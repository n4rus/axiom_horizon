#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

/// A candidate implementation in the Darwin Archive.
/// Each candidate is a patch/diff with associated fitness metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    /// Unique identifier (hash of patch)
    pub id: String,
    /// Human-readable description of the change
    pub description: String,
    /// The patch/diff as a string
    pub patch: String,
    /// Fitness score (higher = better)
    pub fitness: f32,
    /// Components of fitness score
    pub fitness_components: FitnessComponents,
    /// Generation when this candidate was created
    pub generation: usize,
    /// Parent candidate ID (for lineage)
    pub parent_id: Option<String>,
    /// Timestamp of creation
    pub timestamp: u64,
}

/// Components of fitness score
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FitnessComponents {
    /// Performance on held-out tasks (accuracy, F1, etc.)
    pub task_performance: f32,
    /// Inverse of cycle time (faster = better)
    pub speed: f32,
    /// Memory efficiency (lower peak RSS = better)
    pub memory_efficiency: f32,
    /// VFE on validation (lower = better)
    pub vfe: f32,
    /// Novelty vs existing archive (higher = more novel)
    pub novelty: f32,
}

impl FitnessComponents {
    pub fn aggregate(&self) -> f32 {
        0.4 * self.task_performance
            + 0.2 * self.speed
            + 0.1 * self.memory_efficiency
            + 0.2 * (1.0 - self.vfe.min(1.0))
            + 0.1 * self.novelty
    }
}

/// The Darwin Archive — population of candidate implementations
pub struct DarwinArchive {
    /// All candidates, sorted by fitness (best first)
    pub(crate) candidates: Vec<Candidate>,
    /// Minimum fitness threshold for inclusion
    threshold: f32,
    /// Maximum population size
    max_population: usize,
    /// Current generation counter
    generation: usize,
    /// Archive file path
    archive_path: String,
}

impl DarwinArchive {
    /// Access candidates (sorted best-first)
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Mutable access to candidates (for in-place fitness updates)
    pub fn candidates_mut(&mut self) -> &mut Vec<Candidate> {
        &mut self.candidates
    }

    /// Access generation counter
    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn new(archive_path: &str, threshold: f32, max_population: usize) -> Self {
        let mut archive = Self {
            candidates: Vec::new(),
            threshold,
            max_population,
            generation: 0,
            archive_path: archive_path.to_string(),
        };
        archive.load();
        archive
    }

    /// Add a new candidate to the archive
    pub fn add(&mut self, mut candidate: Candidate) -> bool {
        candidate.fitness = candidate.fitness_components.aggregate();
        candidate.generation = self.generation;

        if candidate.fitness < self.threshold {
            return false;
        }

        if self.is_duplicate(&candidate) {
            return false;
        }

        self.candidates.push(candidate);
        self.cull();
        self.save();
        true
    }

    fn is_duplicate(&self, candidate: &Candidate) -> bool {
        for existing in &self.candidates {
            let similarity = self.patch_similarity(&candidate.patch, &existing.patch);
            if similarity > 0.9 {
                return true;
            }
        }
        false
    }

    fn patch_similarity(&self, a: &str, b: &str) -> f32 {
        let lines_a: std::collections::HashSet<_> = a.lines().collect();
        let lines_b: std::collections::HashSet<_> = b.lines().collect();
        let intersection = lines_a.intersection(&lines_b).count();
        let union = lines_a.union(&lines_b).count();
        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }

    pub(crate) fn cull(&mut self) {
        self.candidates.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        if self.candidates.len() > self.max_population {
            self.candidates.truncate(self.max_population);
        }
        if let Some(worst) = self.candidates.last() {
            self.threshold = worst.fitness;
        }
    }

    /// Promote the best candidate for self-modification
    pub fn promote_best(&mut self) -> Option<Candidate> {
        if self.candidates.is_empty() {
            return None;
        }
        let best = self.candidates[0].clone();
        self.generation += 1;
        Some(best)
    }

    pub fn get_above(&self, threshold: f32) -> Vec<Candidate> {
        self.candidates.iter()
            .filter(|c| c.fitness >= threshold)
            .cloned()
            .collect()
    }

    pub fn stats(&self) -> ArchiveStats {
        let avg_fitness = if self.candidates.is_empty() { 0.0 } else {
            self.candidates.iter().map(|c| c.fitness).sum::<f32>() / self.candidates.len() as f32
        };
        ArchiveStats {
            population: self.candidates.len(),
            generation: self.generation,
            best_fitness: self.candidates.first().map(|c| c.fitness).unwrap_or(0.0),
            avg_fitness,
            threshold: self.threshold,
        }
    }

    fn load(&mut self) {
        if Path::new(&self.archive_path).exists() {
            if let Ok(raw) = fs::read_to_string(&self.archive_path) {
                if let Ok(data) = serde_json::from_str::<ArchiveData>(&raw) {
                    self.candidates = data.candidates;
                    self.generation = data.generation;
                    self.threshold = data.threshold;
                }
            }
        }
    }

    pub(crate) fn save(&self) {
        let data = ArchiveData {
            candidates: self.candidates.clone(),
            generation: self.generation,
            threshold: self.threshold,
        };
        if let Ok(json) = serde_json::to_string_pretty(&data) {
            if let Some(parent) = Path::new(&self.archive_path).parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(&self.archive_path, json);
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ArchiveData {
    candidates: Vec<Candidate>,
    generation: usize,
    threshold: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArchiveStats {
    pub population: usize,
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub threshold: f32,
}

/// Genetic operators for generating new candidates
pub mod genetic {
    use super::*;
    use rand::Rng;

    /// Mutation: small random changes to a patch
    pub fn mutate(patch: &str, mutation_rate: f32) -> String {
        let mut rng = rand::thread_rng();
        let lines: Vec<String> = patch.lines().map(|s| s.to_string()).collect();
        let mut result: Vec<String> = Vec::new();
        for line in lines {
            if rng.gen::<f32>() < mutation_rate {
                match rng.gen_range(0..3) {
                    0 => { /* delete - skip */ }
                    1 => { // insert random comment
                        result.push(line);
                        result.push(format!("// mutated at generation {}", rand::random::<u32>()));
                    }
                    2 => { // replace with variant
                        result.push(format!("{} // mutated", line));
                    }
                    _ => result.push(line),
                }
            } else {
                result.push(line);
            }
        }
        result.join("\n")
    }

    /// Crossover: combine two patches
    pub fn crossover(a: &str, b: &str) -> String {
        let lines_a: Vec<&str> = a.lines().collect();
        let lines_b: Vec<&str> = b.lines().collect();
        let mid_a = lines_a.len() / 2;
        let mid_b = lines_b.len() / 2;
        let mut result: Vec<&str> = Vec::new();
        result.extend(&lines_a[..mid_a]);
        result.extend(&lines_b[mid_b..]);
        result.join("\n")
    }

    /// Generate new candidate by mutating the best
    pub fn generate_from_best(archive: &DarwinArchive, mutation_rate: f32) -> Option<Candidate> {
        let best = archive.candidates.first()?;
        let mut rng = rand::thread_rng();
        let new_patch = mutate(&best.patch, mutation_rate);
        Some(Candidate {
            id: format!("cand_{}", rng.gen::<u32>()),
            description: format!("Mutation of {}", best.id),
            patch: new_patch,
            fitness: 0.0,
            fitness_components: FitnessComponents::default(),
            generation: archive.generation + 1,
            parent_id: Some(best.id.clone()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }
}

/// A/B Testing framework for candidate evaluation
pub mod ab_test {
    use super::*;
    use std::time::Instant;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ABTestResult {
        pub candidate_id: String,
        pub baseline_metrics: TestMetrics,
        pub candidate_metrics: TestMetrics,
        pub winner: Winner,
        pub confidence: f32,
        pub duration_secs: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct TestMetrics {
        pub compile_success: bool,
        pub compile_time_secs: f64,
        pub binary_size_mb: f32,
        pub test_pass_rate: f32,
        pub test_count: usize,
        pub vfe: f32,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Winner {
        Candidate,
        Baseline,
        Inconclusive,
    }

    /// Run cargo build and measure metrics
    fn run_cargo_build() -> Result<TestMetrics, String> {
        let start = Instant::now();
        let output = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(".")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| e.to_string())?;
        
        let compile_time = start.elapsed().as_secs_f64();
        let compile_success = output.status.success();
        
        if !compile_success {
            return Ok(TestMetrics {
                compile_success: false,
                compile_time_secs: compile_time,
                ..Default::default()
            });
        }

        // Find binary size
        let binary_size = fs::read_dir("target/release")
            .ok()
            .and_then(|entries| entries
                .filter_map(|e| e.ok())
                .find(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
                .and_then(|e| fs::metadata(e.path()).ok())
                .map(|m| m.len() as f32 / 1_000_000.0)
            )
            .unwrap_or(0.0);

        Ok(TestMetrics {
            compile_success: true,
            compile_time_secs: compile_time,
            binary_size_mb: binary_size,
            ..Default::default()
        })
    }

    fn run_cargo_test() -> Result<(f32, usize), String> {
        let output = Command::new("cargo")
            .args(["test", "--", "--test-threads=1"])
            .current_dir(".")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| e.to_string())?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut passed = 0;
        let mut total = 0;
        
        for line in stdout.lines() {
            if line.contains("test ") && (line.contains("... ok") || line.contains("... FAILED")) {
                total += 1;
                if line.contains("... ok") {
                    passed += 1;
                }
            }
        }
        
        let pass_rate = if total > 0 { passed as f32 / total as f32 } else { 0.0 };
        Ok((pass_rate, total))
    }

    /// Run full test suite and collect metrics
    pub async fn run_test_suite() -> TestMetrics {
        let mut metrics = run_cargo_build().unwrap_or_default();
        
        if metrics.compile_success {
            if let Ok((pass_rate, count)) = run_cargo_test() {
                metrics.test_pass_rate = pass_rate;
                metrics.test_count = count;
            }
        }
        
        metrics
    }

    /// Apply a patch to the codebase
    pub fn apply_patch(base_dir: &Path, patch: &str) -> Result<(), String> {
        let patch_file = base_dir.join(".candidate.patch");
        fs::write(&patch_file, patch).map_err(|e| e.to_string())?;
        
        let output = Command::new("git")
            .args(["apply", "--whitespace=nowarn", patch_file.to_str().unwrap()])
            .current_dir(base_dir)
            .output()
            .map_err(|e| e.to_string())?;
        
        let _ = fs::remove_file(&patch_file);
        
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }
        Ok(())
    }

    /// Revert a patch
    pub fn revert_patch(base_dir: &Path) -> Result<(), String> {
        let output = Command::new("git")
            .args(["checkout", "--", "."])
            .current_dir(base_dir)
            .output()
            .map_err(|e| e.to_string())?;
        
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }
        Ok(())
    }

    /// Run A/B test: compare candidate vs baseline
    pub async fn run_ab_test(
        candidate_patch: &str,
        _test_tasks: &[TestTask],
    ) -> Result<ABTestResult, String> {
        let start = Instant::now();
        
        // 1. Get baseline metrics (current code)
        let baseline_metrics = run_test_suite().await;
        
        // 2. Apply candidate patch
        apply_patch(Path::new("."), candidate_patch)?;
        
        // 3. Run candidate tests
        let candidate_metrics = run_test_suite().await;
        
        // 3. Revert patch
        revert_patch(Path::new("."))?;
        
        // 4. Determine winner
        let (winner, confidence) = determine_winner(&baseline_metrics, &candidate_metrics);
        
        Ok(ABTestResult {
            candidate_id: String::new(),
            baseline_metrics,
            candidate_metrics,
            winner,
            confidence,
            duration_secs: start.elapsed().as_secs_f64(),
        })
    }

    fn determine_winner(baseline: &TestMetrics, candidate: &TestMetrics) -> (Winner, f32) {
        let mut score = 0.0;

        if candidate.compile_success && !baseline.compile_success {
            score += 1.0;
        } else if !candidate.compile_success && baseline.compile_success {
            score -= 1.0;
        }
        
        if candidate.test_pass_rate > baseline.test_pass_rate {
            score += candidate.test_pass_rate - baseline.test_pass_rate;
        } else if candidate.test_pass_rate < baseline.test_pass_rate {
            score -= baseline.test_pass_rate - candidate.test_pass_rate;
        }
        if candidate.compile_time_secs < baseline.compile_time_secs {
            let diff = (baseline.compile_time_secs - candidate.compile_time_secs) as f32;
            score += diff / baseline.compile_time_secs.max(1.0) as f32;
        }
        if candidate.binary_size_mb < baseline.binary_size_mb && baseline.binary_size_mb > 0.0 {
            score += (baseline.binary_size_mb - candidate.binary_size_mb) / baseline.binary_size_mb;
        }
        
        if score > 0.05 {
            (Winner::Candidate, score.min(0.95))
        } else if score < -0.05 {
            (Winner::Baseline, (-score).min(0.95))
        } else {
            (Winner::Inconclusive, 0.5)
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TestTask {
        pub name: String,
        pub input: String,
        pub expected_output: Option<String>,
        pub category: String,
    }
}