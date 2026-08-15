//! Phase 4.2: Goal Decomposition — break complex tasks into subtasks automatically.
//! From AGI_PLAN.md §4.2: "Break complex tasks into sub-tasks automatically.
//! The agent generates sub-tasks, executes each, checks success, re-plans if failed."

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Status of a subtask.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubtaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed { reason: String },
    Skipped { reason: String },
}

/// A single subtask in a decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    pub id: usize,
    pub label: String,
    pub description: String,
    pub status: SubtaskStatus,
    pub subtasks: Vec<Subtask>,
}

impl Subtask {
    pub fn new(id: usize, label: &str, description: &str) -> Self {
        Subtask {
            id,
            label: label.to_string(),
            description: description.to_string(),
            status: SubtaskStatus::Pending,
            subtasks: Vec::new(),
        }
    }

    pub fn is_done(&self) -> bool {
        match &self.status {
            SubtaskStatus::Completed | SubtaskStatus::Skipped { .. } => true,
            SubtaskStatus::Failed { .. } => false,
            SubtaskStatus::Pending | SubtaskStatus::InProgress => {
                // Done if all children are done
                !self.subtasks.is_empty() && self.subtasks.iter().all(|s| s.is_done())
            }
        }
    }

    #[allow(dead_code)]
    pub fn mark_completed(&mut self) {
        self.status = SubtaskStatus::Completed;
    }

    #[allow(dead_code)]
    pub fn mark_failed(&mut self, reason: &str) {
        self.status = SubtaskStatus::Failed { reason: reason.to_string() };
    }
}

/// A decomposed goal with its subtask tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalPlan {
    pub goal: String,
    pub subtasks: Vec<Subtask>,
    pub max_depth: usize,
}

impl GoalPlan {
    #[allow(dead_code)]
    pub fn is_complete(&self) -> bool {
        self.subtasks.iter().all(|s| s.is_done())
    }

    pub fn progress(&self) -> f32 {
        if self.subtasks.is_empty() { return 1.0; }
        let done = self.subtasks.iter().filter(|s| s.is_done()).count() as f32;
        done / self.subtasks.len() as f32
    }

    #[allow(dead_code)]
    pub fn next_pending(&self) -> Option<&Subtask> {
        for s in &self.subtasks {
            if !s.is_done() {
                return Some(s);
            }
        }
        None
    }

    pub fn summary(&self) -> String {
        let total = self.subtasks.len();
        let done = self.subtasks.iter().filter(|s| s.is_done()).count();
        let failed = self.subtasks.iter()
            .filter(|s| matches!(s.status, SubtaskStatus::Failed { .. }))
            .count();
        format!(
            "{}/{} done, {} failed, progress: {:.0}%",
            done, total, failed, self.progress() * 100.0
        )
    }
}

/// Goal decomposer — breaks high-level goals into subtasks using rule-based patterns.
pub struct GoalDecomposer {
    max_depth: usize,
}

impl GoalDecomposer {
    pub fn new() -> Self {
        GoalDecomposer { max_depth: 3 }
    }

    #[allow(dead_code)]
    pub fn with_max_depth(max_depth: usize) -> Self {
        GoalDecomposer { max_depth }
    }

    /// Decompose a goal string into a plan with subtasks.
    pub fn decompose(&self, goal: &str) -> GoalPlan {
        let subtasks = self.decompose_inner(goal, 0);
        GoalPlan {
            goal: goal.to_string(),
            subtasks,
            max_depth: self.max_depth,
        }
    }

    fn decompose_inner(&self, goal: &str, depth: usize) -> Vec<Subtask> {
        if depth >= self.max_depth {
            return vec![Subtask::new(0, "execute", goal)];
        }

        let lower = goal.to_lowercase();
        let mut subtasks = Vec::new();
        let mut id = 0;

        // Pattern: "build" / "create" / "implement"
        if lower.contains("build") || lower.contains("create") || lower.contains("implement") {
            subtasks.push(Subtask::new(id, "research", "Research best approach and requirements"));
            id += 1;
            subtasks.push(Subtask::new(id, "design", "Design architecture and interfaces"));
            id += 1;
            subtasks.push(Subtask::new(id, "implement", &format!("Implement core: {goal}")));
            id += 1;
            subtasks.push(Subtask::new(id, "test", "Write tests and verify"));
            id += 1;
            subtasks.push(Subtask::new(id, "integrate", "Integrate and validate end-to-end"));
        }
        // Pattern: "fix" / "repair" / "debug"
        else if lower.contains("fix") || lower.contains("repair") || lower.contains("debug") {
            subtasks.push(Subtask::new(id, "reproduce", "Reproduce the issue"));
            id += 1;
            subtasks.push(Subtask::new(id, "diagnose", "Diagnose root cause"));
            id += 1;
            subtasks.push(Subtask::new(id, "fix", "Apply fix"));
            id += 1;
            subtasks.push(Subtask::new(id, "verify", "Verify fix resolves the issue"));
        }
        // Pattern: "research" / "investigate" / "explore"
        else if lower.contains("research") || lower.contains("investigate") || lower.contains("explore") {
            subtasks.push(Subtask::new(id, "define-scope", "Define research scope and questions"));
            id += 1;
            subtasks.push(Subtask::new(id, "gather", "Gather information from sources"));
            id += 1;
            subtasks.push(Subtask::new(id, "analyze", "Analyze findings"));
            id += 1;
            subtasks.push(Subtask::new(id, "summarize", "Summarize and produce deliverable"));
        }
        // Pattern: "test" / "verify" / "validate"
        else if lower.contains("test") || lower.contains("verify") || lower.contains("validate") {
            subtasks.push(Subtask::new(id, "define-cases", "Define test cases"));
            id += 1;
            subtasks.push(Subtask::new(id, "setup", "Set up test environment"));
            id += 1;
            subtasks.push(Subtask::new(id, "execute", "Execute tests"));
            id += 1;
            subtasks.push(Subtask::new(id, "report", "Report results"));
        }
        // Pattern: "deploy" / "release" / "ship"
        else if lower.contains("deploy") || lower.contains("release") || lower.contains("ship") {
            subtasks.push(Subtask::new(id, "prepare", "Prepare build and config"));
            id += 1;
            subtasks.push(Subtask::new(id, "deploy", "Deploy to target environment"));
            id += 1;
            subtasks.push(Subtask::new(id, "smoke-test", "Smoke test deployment"));
            id += 1;
            subtasks.push(Subtask::new(id, "monitor", "Monitor and confirm stability"));
        }
        // Pattern: "improve" / "optimize" / "enhance"
        else if lower.contains("improve") || lower.contains("optimize") || lower.contains("enhance") {
            subtasks.push(Subtask::new(id, "benchmark", "Benchmark current performance"));
            id += 1;
            subtasks.push(Subtask::new(id, "identify", "Identify improvement opportunities"));
            id += 1;
            subtasks.push(Subtask::new(id, "implement", "Implement improvements"));
            id += 1;
            subtasks.push(Subtask::new(id, "re-benchmark", "Re-benchmark and compare"));
        }
        // Pattern: "learn" / "study" / "understand"
        else if lower.contains("learn") || lower.contains("study") || lower.contains("understand") {
            subtasks.push(Subtask::new(id, "find-materials", "Find learning materials"));
            id += 1;
            subtasks.push(Subtask::new(id, "read", "Read and take notes"));
            id += 1;
            subtasks.push(Subtask::new(id, "practice", "Practice with examples"));
            id += 1;
            subtasks.push(Subtask::new(id, "synthesize", "Synthesize understanding"));
        }
        // Generic fallback
        else {
            subtasks.push(Subtask::new(id, "plan", &format!("Plan approach for: {goal}")));
            id += 1;
            subtasks.push(Subtask::new(id, "execute", &format!("Execute: {goal}")));
            id += 1;
            subtasks.push(Subtask::new(id, "verify", "Verify completion"));
        }

        // Recurse into each subtask if depth allows
        if depth + 1 < self.max_depth {
            for sub in &mut subtasks {
                sub.subtasks = self.decompose_inner(&sub.description, depth + 1);
            }
        }

        subtasks
    }
}

impl Default for GoalDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Flatten a plan's subtask tree into a VecDeque for iterative execution.
#[allow(dead_code)]
pub fn flatten_plan(plan: &GoalPlan) -> VecDeque<&Subtask> {
    let mut queue = VecDeque::new();
    for s in &plan.subtasks {
        flatten_subtask(s, &mut queue);
    }
    queue
}

#[allow(dead_code)]
fn flatten_subtask<'a>(sub: &'a Subtask, queue: &mut VecDeque<&'a Subtask>) {
    if sub.subtasks.is_empty() {
        queue.push_back(sub);
    } else {
        for child in &sub.subtasks {
            flatten_subtask(child, queue);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_decomposition() {
        let decomposer = GoalDecomposer::new();
        let plan = decomposer.decompose("Build a web scraper");
        assert!(plan.subtasks.len() >= 4);
        assert!(!plan.is_complete());
    }

    #[test]
    fn test_fix_pattern() {
        let decomposer = GoalDecomposer::new();
        let plan = decomposer.decompose("Fix the authentication bug");
        assert_eq!(plan.subtasks.len(), 4);
        assert_eq!(plan.subtasks[0].label, "reproduce");
        assert_eq!(plan.subtasks[1].label, "diagnose");
    }

    #[test]
    fn test_research_pattern() {
        let decomposer = GoalDecomposer::new();
        let plan = decomposer.decompose("Research quantum computing applications");
        assert_eq!(plan.subtasks.len(), 4);
        assert_eq!(plan.subtasks[0].label, "define-scope");
    }

    #[test]
    fn test_progress_tracking() {
        let decomposer = GoalDecomposer::with_max_depth(1);
        let mut plan = decomposer.decompose("Build something");
        assert_eq!(plan.progress(), 0.0);
        plan.subtasks[0].mark_completed();
        assert!(plan.progress() > 0.0);
        assert!(!plan.is_complete());
    }

    #[test]
    fn test_nested_decomposition() {
        let decomposer = GoalDecomposer::with_max_depth(2);
        let plan = decomposer.decompose("Build a REST API");
        // Each top-level subtask should have children
        for s in &plan.subtasks {
            assert!(!s.subtasks.is_empty());
        }
    }

    #[test]
    fn test_generic_fallback() {
        let decomposer = GoalDecomposer::new();
        let plan = decomposer.decompose("Do something unspecified");
        assert!(plan.subtasks.len() >= 3);
    }

    #[test]
    fn test_next_pending() {
        let decomposer = GoalDecomposer::new();
        let plan = decomposer.decompose("Fix a bug");
        let next = plan.next_pending();
        assert!(next.is_some());
        assert_eq!(next.unwrap().label, "reproduce");
    }

    #[test]
    fn test_flatten() {
        let decomposer = GoalDecomposer::new();
        let plan = decomposer.decompose("Build and test");
        let flat = flatten_plan(&plan);
        assert!(!flat.is_empty());
    }

    #[test]
    fn test_summary() {
        let decomposer = GoalDecomposer::new();
        let plan = decomposer.decompose("Build X");
        let s = plan.summary();
        assert!(s.contains("progress"));
    }

    #[test]
    fn test_max_depth_limit() {
        let decomposer = GoalDecomposer::with_max_depth(1);
        let plan = decomposer.decompose("Build deep thing");
        for s in &plan.subtasks {
            assert!(s.subtasks.is_empty());
        }
    }
}
