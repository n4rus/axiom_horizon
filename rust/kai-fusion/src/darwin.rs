#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use syn::{parse_file, visit_mut::VisitMut, File, Item, Stmt, Expr, BinOp, ExprBinary, ExprLit, Lit, Block, Pat, Type, ReturnType, FnArg, Signature, Generics, WhereClause, WherePredicate, TraitBound, TypeParamBound, Lifetime, GenericParam, AngleBracketedGenericArguments, GenericArgument, PathArguments, PathSegment, Ident, Token, punctuated::Punctuated};

/// AST-level mutation operators for Rust source code.
/// Uses syn/quote for type-safe Rust AST manipulation.
pub mod ast_mutator {
    use super::*;
    use rand::seq::SliceRandom;

    /// A mutation operation that can be applied to a Rust file
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Mutation {
        /// Insert a new statement into a function body
        InsertStmt { func_name: String, stmt: String, position: usize },
        /// Delete a statement
        DeleteStmt { func_name: String, index: usize },
        /// Replace a statement
        ReplaceStmt { func_name: String, index: usize, new_stmt: String },
        /// Replace a literal value
        ReplaceLit { func_name: String, old_lit: String, new_lit: String },
        /// Replace a binary operator
        ReplaceBinOp { func_name: String, old_op: String, new_op: BinOpKind },
        /// Add a conditional branch
        AddIfBranch { func_name: String, condition: String, then_block: String, else_block: Option<String>, position: usize },
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub enum BinOpKind {
        Add, Sub, Mul, Div, Rem,
        And, Or, BitXor, BitAnd, BitOr,
        Shl, Shr,
        Eq, Ne, Lt, Le, Gt, Ge,
    }

    impl BinOpKind {
        pub fn all() -> Vec<BinOpKind> {
            vec![
                BinOpKind::Add, BinOpKind::Sub, BinOpKind::Mul, BinOpKind::Div, BinOpKind::Rem,
                BinOpKind::And, BinOpKind::Or, BinOpKind::BitXor, BinOpKind::BitAnd, BinOpKind::BitOr,
                BinOpKind::Shl, BinOpKind::Shr,
                BinOpKind::Eq, BinOpKind::Ne, BinOpKind::Lt, BinOpKind::Le, BinOpKind::Gt, BinOpKind::Ge,
            ]
        }

        pub fn to_token(&self) -> TokenStream {
            match self {
                BinOpKind::Add => quote! { + },
                BinOpKind::Sub => quote! { - },
                BinOpKind::Mul => quote! { * },
                BinOpKind::Div => quote! { / },
                BinOpKind::Rem => quote! { % },
                BinOpKind::And => quote! { && },
                BinOpKind::Or => quote! { || },
                BinOpKind::BitXor => quote! { ^ },
                BinOpKind::BitAnd => quote! { & },
                BinOpKind::BitOr => quote! { | },
                BinOpKind::Shl => quote! { << },
                BinOpKind::Shr => quote! { >> },
                BinOpKind::Eq => quote! { == },
                BinOpKind::Ne => quote! { != },
                BinOpKind::Lt => quote! { < },
                BinOpKind::Le => quote! { <= },
                BinOpKind::Gt => quote! { > },
                BinOpKind::Ge => quote! { >= },
            }
        }
    }

    /// A location in a function body where mutations can be applied
    #[derive(Debug, Clone)]
    pub struct MutationSite {
        pub func_name: String,
        pub kind: SiteKind,
        pub index: usize,
    }

    #[derive(Debug, Clone)]
    pub enum SiteKind {
        Statement { preview: String },
        Literal { value: String },
        BinOp { op: String, left: String, right: String },
    }

    /// Collect all mutation sites in a Rust file by parsing function bodies
    pub fn collect_sites(source: &str) -> Result<Vec<MutationSite>, syn::Error> {
        let file = parse_file(source)?;
        let mut sites = Vec::new();

        for item in &file.items {
            if let Item::Fn(func) = item {
                let func_name = func.sig.ident.to_string();
                let block = &func.block;

                for (i, stmt) in block.stmts.iter().enumerate() {
                    let preview = quote!(#stmt).to_string();
                    if preview.len() > 80 {
                        sites.push(MutationSite {
                            func_name: func_name.clone(),
                            kind: SiteKind::Statement { preview: preview[..80].to_string() + "..." },
                            index: i,
                        });
                    } else {
                        sites.push(MutationSite {
                            func_name: func_name.clone(),
                            kind: SiteKind::Statement { preview },
                            index: i,
                        });
                    }

                    collect_literals_and_binops(&mut vec![], &func_name, stmt, i, &mut vec![]);
                }
            }
        }
        Ok(sites)
    }

    fn collect_literals_and_binops(sites: &mut Vec<MutationSite>, func_name: &str, stmt: &Stmt, stmt_idx: usize, _path: &mut Vec<usize>) {
        match stmt {
            Stmt::Expr(expr, _semi) => {
                collect_in_expr(sites, func_name, expr, stmt_idx);
            }
            Stmt::Local(local) => {
                if let Some(init) = &local.init {
                    collect_in_expr(sites, func_name, &init.expr, stmt_idx);
                }
            }
            Stmt::Item(_) => {}
            Stmt::Macro(_) => {}
        }
    }

    fn collect_in_expr(sites: &mut Vec<MutationSite>, func_name: &str, expr: &Expr, stmt_idx: usize) {
        match expr {
            Expr::Lit(lit) => {
                if let Lit::Str(s) = &lit.lit {
                    sites.push(MutationSite {
                        func_name: func_name.to_string(),
                        kind: SiteKind::Literal { value: s.value() },
                        index: stmt_idx,
                    });
                } else if let Lit::Int(i) = &lit.lit {
                    sites.push(MutationSite {
                        func_name: func_name.to_string(),
                        kind: SiteKind::Literal { value: i.base10_digits().to_string() },
                        index: stmt_idx,
                    });
                } else if let Lit::Float(f) = &lit.lit {
                    sites.push(MutationSite {
                        func_name: func_name.to_string(),
                        kind: SiteKind::Literal { value: f.base10_digits().to_string() },
                        index: stmt_idx,
                    });
                }
            }
            Expr::Binary(bin) => {
                let left_str = quote!(#bin.left).to_string();
                let right_str = quote!(#bin.right).to_string();
                sites.push(MutationSite {
                    func_name: func_name.to_string(),
                    kind: SiteKind::BinOp { op: bin.op.to_token_stream().to_string(), left: left_str, right: right_str },
                    index: stmt_idx,
                });
                collect_in_expr(sites, func_name, &bin.left, stmt_idx);
                collect_in_expr(sites, func_name, &bin.right, stmt_idx);
            }
            Expr::Block(block) => {
                for inner_stmt in &block.block.stmts {
                    collect_literals_and_binops(sites, func_name, inner_stmt, stmt_idx, &mut vec![]);
                }
            }
            Expr::If(if_expr) => {
                collect_in_expr(sites, func_name, &if_expr.cond, stmt_idx);
                for inner_stmt in &if_expr.then_branch.stmts {
                    collect_literals_and_binops(sites, func_name, inner_stmt, stmt_idx, &mut vec![]);
                }
                if let Some((_, else_expr)) = &if_expr.else_branch {
                    collect_in_expr(sites, func_name, else_expr, stmt_idx);
                }
            }
            Expr::Match(m) => {
                collect_in_expr(sites, func_name, &m.expr, stmt_idx);
                for arm in &m.arms {
                    if let Expr::Block(b) = &*arm.body {
                        for inner_stmt in &b.block.stmts {
                            collect_literals_and_binops(sites, func_name, inner_stmt, stmt_idx, &mut vec![]);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Apply a mutation to a source file
    pub fn apply_mutation(source: &str, mutation: &Mutation) -> Result<String, String> {
        let mut file = parse_file(source).map_err(|e| format!("parse: {e}"))?;

        for item in &mut file.items {
            if let Item::Fn(func) = item {
                let func_name = func.sig.ident.to_string();
                let block = &mut func.block;

                match mutation {
                    Mutation::InsertStmt { func_name: target_func, stmt, position } => {
                        if func_name == *target_func {
                            let new_stmt: Stmt = syn::parse_str(stmt).map_err(|e| format!("parse stmt: {e}"))?;
                            if *position <= block.stmts.len() {
                                block.stmts.insert(*position, new_stmt);
                            }
                        }
                    }
                    Mutation::DeleteStmt { func_name: target_func, index } => {
                        if func_name == *target_func && *index < block.stmts.len() {
                            block.stmts.remove(*index);
                        }
                    }
                    Mutation::ReplaceStmt { func_name: target_func, index, new_stmt } => {
                        if func_name == *target_func && *index < block.stmts.len() {
                            let new_stmt_parsed: Stmt = syn::parse_str(new_stmt).map_err(|e| format!("parse stmt: {e}"))?;
                            block.stmts[*index] = new_stmt_parsed;
                        }
                    }
                    Mutation::ReplaceLit { func_name: target_func, old_lit, new_lit } => {
                        if func_name == *target_func {
                            replace_literals_in_block(block, old_lit, new_lit);
                        }
                    }
                    Mutation::ReplaceBinOp { func_name: target_func, old_op, new_op } => {
                        if func_name == *target_func {
                            replace_binops_in_block(block, old_op, &new_op.to_token());
                        }
                    }
                    Mutation::AddIfBranch { func_name: target_func, condition, then_block, else_block, position } => {
                        if func_name == *target_func {
                            let cond_expr: Expr = syn::parse_str(condition).map_err(|e| format!("parse condition: {e}"))?;
                            let then_block_parsed: Block = syn::parse_str(&format!("{{ {} }}", then_block)).map_err(|e| format!("parse then: {e}"))?;
                            let else_branch = if let Some(eb) = else_block {
                                let else_block_parsed: Block = syn::parse_str(&format!("{{ {} }}", eb)).map_err(|e| format!("parse else: {e}"))?;
                                Some((Token![else](proc_macro2::Span::call_site()), Box::new(Expr::Block(syn::ExprBlock { attrs: Vec::new(), label: None, block: else_block_parsed }))))
                            } else { None };

                            let if_expr = Expr::If(syn::ExprIf {
                                if_token: Token![if](proc_macro2::Span::call_site()),
                                attrs: Vec::new(),
                                cond: Box::new(cond_expr),
                                then_branch: then_block_parsed,
                                else_branch,
                            });
                            block.stmts.insert(*position, Stmt::Expr(if_expr, None));
                        }
                    }
                }
            }
        }

        Ok(quote!(#file).to_string())
    }

    fn replace_literals_in_block(block: &mut Block, old_lit: &str, new_lit: &str) {
        for stmt in &mut block.stmts {
            replace_literals_in_stmt(stmt, old_lit, new_lit);
        }
    }

    fn replace_literals_in_stmt(stmt: &mut Stmt, old_lit: &str, new_lit: &str) {
        match stmt {
            Stmt::Expr(expr, _semi) => replace_literals_in_expr(expr, old_lit, new_lit),
            Stmt::Local(local) => {
                if let Some(init) = &mut local.init {
                    replace_literals_in_expr(&mut init.expr, old_lit, new_lit);
                }
            }
            Stmt::Item(_) => {}
            Stmt::Macro(_) => {}
        }
    }

    fn replace_literals_in_expr(expr: &mut Expr, old_lit: &str, new_lit: &str) {
        match expr {
            Expr::Lit(lit) => {
                let current = quote!(#lit).to_string();
                if current.contains(old_lit) {
                    if let Ok(new_lit_parsed) = syn::parse_str::<Expr>(new_lit) {
                        *expr = new_lit_parsed;
                    }
                }
            }
            Expr::Binary(bin) => {
                replace_literals_in_expr(&mut bin.left, old_lit, new_lit);
                replace_literals_in_expr(&mut bin.right, old_lit, new_lit);
            }
            Expr::Block(block) => {
                for stmt in &mut block.block.stmts {
                    replace_literals_in_stmt(stmt, old_lit, new_lit);
                }
            }
            Expr::If(if_expr) => {
                replace_literals_in_expr(&mut if_expr.cond, old_lit, new_lit);
                for stmt in &mut if_expr.then_branch.stmts {
                    replace_literals_in_stmt(stmt, old_lit, new_lit);
                }
                if let Some((_, else_expr)) = &mut if_expr.else_branch {
                    replace_literals_in_expr(else_expr, old_lit, new_lit);
                }
            }
            _ => {}
        }
    }

    fn replace_binops_in_block(block: &mut Block, old_op: &str, new_op: &TokenStream) {
        for stmt in &mut block.stmts {
            replace_binops_in_stmt(stmt, old_op, new_op);
        }
    }

    fn replace_binops_in_stmt(stmt: &mut Stmt, old_op: &str, new_op: &TokenStream) {
        match stmt {
            Stmt::Expr(expr, _semi) => replace_binops_in_expr(expr, old_op, new_op),
            Stmt::Local(local) => {
                if let Some(init) = &mut local.init {
                    replace_binops_in_expr(&mut init.expr, old_op, new_op);
                }
            }
            Stmt::Item(_) => {}
            Stmt::Macro(_) => {}
        }
    }

    fn replace_binops_in_expr(expr: &mut Expr, old_op: &str, new_op: &TokenStream) {
        match expr {
            Expr::Binary(bin) => {
                let current_op = bin.op.to_token_stream().to_string();
                if current_op == old_op {
                    bin.op = syn::parse2(new_op.clone()).unwrap_or(bin.op.clone());
                }
                replace_binops_in_expr(&mut bin.left, old_op, new_op);
                replace_binops_in_expr(&mut bin.right, old_op, new_op);
            }
            Expr::Block(block) => {
                for stmt in &mut block.block.stmts {
                    replace_binops_in_stmt(stmt, old_op, new_op);
                }
            }
            Expr::If(if_expr) => {
                replace_binops_in_expr(&mut if_expr.cond, old_op, new_op);
                for stmt in &mut if_expr.then_branch.stmts {
                    replace_binops_in_stmt(stmt, old_op, new_op);
                }
                if let Some((_, else_expr)) = &mut if_expr.else_branch {
                    replace_binops_in_expr(else_expr, old_op, new_op);
                }
            }
            _ => {}
        }
    }

    /// Generate a random mutation for a given file
    pub fn random_mutation(source: &str) -> Result<Mutation, String> {
        let sites = collect_sites(source).map_err(|e| e.to_string())?;
        if sites.is_empty() {
            return Err("no mutation sites".into());
        }
        let mut rng = rand::thread_rng();
        let site = sites.choose(&mut rng).unwrap();

        match &site.kind {
            SiteKind::Statement { .. } => {
                let stmts = vec![
                    "let x = 1;".to_string(),
                    "println!(\"debug\");".to_string(),
                    "if true {}".to_string(),
                ];
                Ok(Mutation::InsertStmt {
                    func_name: site.func_name.clone(),
                    stmt: stmts.choose(&mut rng).unwrap().clone(),
                    position: rng.gen_range(0..=site.index),
                })
            }
            SiteKind::Literal { .. } => {
                Ok(Mutation::ReplaceLit {
                    func_name: site.func_name.clone(),
                    old_lit: "".to_string(),
                    new_lit: format!("\"mutated_{}\"", rng.gen::<u32>()),
                })
            }
            SiteKind::BinOp { .. } => {
                let ops = BinOpKind::all();
                Ok(Mutation::ReplaceBinOp {
                    func_name: site.func_name.clone(),
                    old_op: "".to_string(),
                    new_op: ops.choose(&mut rng).unwrap().clone(),
                })
            }
        }
    }
}

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
    /// Adaptive mutation amplitude (darwin can evolve this rate, hence it is
    /// persisted with the archive, not a compile-time constant).
    mutation_rate: f64,
    /// Best fitness of the previous generation (delta-gen gate reference).
    last_gen_best: f64,
    /// Count of promotions that were strict improvements (parent->child
    /// fitness delta > 0). Metric target: >70% of promoted children improve.
    strict_promotions: u64,
    /// Total promotions attempted (denominator for the strict-improvement
    /// metric).
    total_promotions: u64,
    /// Minimum generation-over-generation fitness delta to allow a promotion.
    min_promotion_delta: f64,
    /// Episodic replay buffer (LAYER 3e): the LAST successful interactions
    /// observed by the world (query→answer→tool→outcome), ingested from the
    /// bridge's chat memory. Persisted so evolution is grounded in the recent
    /// experience stream, not just the static source corpus.
    episodes: Vec<Episode>,
}

/// One replayed interaction: (query → answer) as absorbed by the bridge's
/// chat memory store. `kind` mirrors the Python-side entry kind ("chat"),
/// and `ts` lets us keep the most recent N episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub query: String,
    pub answer: String,
    pub ts: u64,
    /// Provenance: "chat" (live conversation) or "self-report" (bridge ingest).
    pub kind: String,
}

impl Episode {
    pub fn is_successful(&self) -> bool {
        // A successful episode is one with a substantive answer.
        self.answer.trim().len() >= 16
    }
}

fn default_mutation_rate() -> f64 { 1.0 }
fn default_min_delta() -> f64 { 0.05 }
const MAX_REPLAY_EPISODES: usize = 64;

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
            mutation_rate: default_mutation_rate(),
            last_gen_best: 0.0,
            strict_promotions: 0,
            total_promotions: 0,
            min_promotion_delta: default_min_delta(),
            episodes: Vec::new(),
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

    /// Promote the best candidate for self-modification.
    ///
    /// LAYER 3b: promotion is gated by the safety invariants. The best candidate
    /// is only promoted if its patch passes `SafetyGuard::check_darwin_promotion`;
    /// otherwise the next-best safe candidate is chosen. If every candidate is
    /// unsafe, we refuse promotion entirely rather than mutate the mainline.
    pub fn promote_best(&mut self) -> Option<Candidate> {
        if self.candidates.is_empty() {
            return None;
        }
        let guard = crate::safety::SafetyGuard::new();
        // Start at the best and walk down until a safe candidate is found.
        for candidate in &self.candidates {
            match guard.check_darwin_promotion(&candidate.id, &candidate.patch) {
                crate::safety::SafetyCheck::Allowed => {
                    self.generation += 1;
                    return Some(candidate.clone());
                }
                _ => continue, // skip unsafe candidates regardless of fitness
            }
        }
        None // every candidate unsafe -> do not self-modify
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
                    self.mutation_rate = data.mutation_rate;
                    self.last_gen_best = data.last_gen_best;
                    self.strict_promotions = data.strict_promotions;
                    self.total_promotions = data.total_promotions;
                    self.min_promotion_delta = data.min_promotion_delta;
                    self.episodes = data.episodes;
                }
            }
        }
    }

    pub(crate) fn save(&self) {
        let data = ArchiveData {
            candidates: self.candidates.clone(),
            generation: self.generation,
            threshold: self.threshold,
            mutation_rate: self.mutation_rate,
            last_gen_best: self.last_gen_best,
            strict_promotions: self.strict_promotions,
            total_promotions: self.total_promotions,
            min_promotion_delta: self.min_promotion_delta,
            episodes: self.episodes.clone(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&data) {
            if let Some(parent) = Path::new(&self.archive_path).parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(&self.archive_path, json);
        }
    }

    /// Current adaptive mutation amplitude (1.0 = default ±20%).
    pub fn mutation_rate(&self) -> f64 {
        self.mutation_rate
    }

    /// Fitness of the last generation that passed the delta-gen gate.
    pub fn last_gen_fitness(&self) -> f64 {
        self.last_gen_best
    }

    /// Delta-gen gate for the CLI evolution loop.
    ///
    /// Given the best fitness observed this generation and the delta vs the
    /// previous generation, adapt the mutation amplitude (narrow on
    /// improvement, widen on regression/plateau) and only advance the
    /// baseline (`last_gen_best`) — and the generation counter — when the
    /// observed fitness is a strict improvement. Returns whether the
    /// generation improved.
    ///
    /// This is the pressure that keeps recursive self-improvement directed:
    /// the goalposts never move backwards.
    pub fn gate_generation(&mut self, best_fitness: f64, delta: f64) -> bool {
        self.total_promotions += 1;
        self.adapt_mutation_rate(delta);
        if best_fitness > self.last_gen_best + self.min_promotion_delta {
            self.last_gen_best = best_fitness;
            self.generation += 1;
            self.strict_promotions += 1;
            true
        } else {
            false
        }
    }

    /// Adjust the mutation rate from last generation's delta-gen fitness.
    ///
    /// LAYER "fitness-gated evolution": the mutation amplitude is itself a
    /// darwin state. A strict improvement (delta > 0) decays the rate toward
    /// the floor (converge on a good basin); a regression/plateau raises it
    /// (explore more). This replaces the blind fixed-rate search with a
    /// directed one: search narrows where the physics objective improves and
    /// broadens where it doesn't.
    pub fn adapt_mutation_rate(&mut self, delta_gen_fitness: f64) {
        let rate = self.mutation_rate;
        let next = if delta_gen_fitness > 0.0 {
            // improving: trust the neighborhood, shrink the step
            rate * 0.85
        } else {
            // plateau or regression: widen the search
            rate * 1.4
        };
        self.mutation_rate = next.clamp(0.25, 6.0);
    }

    /// Fitness-gated promotion: promote the best candidate only if it would be
    /// a strict improvement over the best fitness of the previous generation.
    ///
    /// Returns `None` when no candidate exceeds `last_gen_best + min_delta`,
    /// or when every candidate is blocked by the safety invariants. This is
    /// the delta-gen gate: it prevents blind mutation and refusing to lift a
    /// regression, so search stays directed.
    pub fn promote_best_gated(&mut self) -> Option<Candidate> {
        let baseline = self.last_gen_best.max(self.threshold as f64);
        self.total_promotions += 1;
        let guard = crate::safety::SafetyGuard::new();
        for candidate in &self.candidates {
            let verdict = guard.check_darwin_promotion(&candidate.id, &candidate.patch);
            if !matches!(verdict, crate::safety::SafetyCheck::Allowed) {
                continue;
            }
            let f = candidate.fitness as f64;
            if f <= baseline + self.min_promotion_delta {
                // Not a strict improvement; keep scanning for a better safe one.
                continue;
            }
            // Strict improvement AND safe -> promote.
            self.generation += 1;
            self.strict_promotions += 1;
            self.last_gen_best = f;
            return Some(candidate.clone());
        }
        None
    }

    /// Fraction of promotions that were strict improvements (target >70%).
    pub fn strict_improvement_ratio(&self) -> f64 {
        if self.total_promotions == 0 {
            return 0.0;
        }
        self.strict_promotions as f64 / self.total_promotions as f64
    }

    // ── Episodic replay (LAYER 3e) ──────────────────────────────────────

    /// Number of episodes currently in the replay buffer.
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Push a replayed episode, dropping only if superseded by a newer one
    /// with identical query+answer (dedup) and capping to the newest N.
    /// Returns true if the episode was actually added.
    pub fn ingest_episode(&mut self, ep: Episode) -> bool {
        if !ep.is_successful() {
            return false;
        }
        // Dedup: identical (query,answer) already present? Don't re-add.
        if self.episodes.iter().any(|e| e.query == ep.query && e.answer == ep.answer) {
            return false;
        }
        self.episodes.push(ep);
        // Keep the newest MAX_REPLAY_EPISODES (by ts, oldest dropped).
        self.episodes.sort_by(|a, b| b.ts.cmp(&a.ts));
        if self.episodes.len() > MAX_REPLAY_EPISODES {
            self.episodes.truncate(MAX_REPLAY_EPISODES);
        }
        self.save();
        true
    }

    /// Most recent N episodes (newest first), optionally filtered to a kind.
    pub fn recent_episodes(&self, n: usize, kind: &str) -> Vec<&Episode> {
        self.episodes.iter()
            .filter(|e| e.kind == kind)
            .take(n)
            .collect()
    }

    /// Serialize the replay buffer into a compact one-line-per-episode
    /// digest. Used by the self-play trainer to seed mutation candidates,
    /// so evolution is grounded in what actually worked in the world.
    pub fn replay_digest(&self, max_episodes: usize, max_chars: usize) -> String {
        let mut out = String::from("// ── kai episodic replay (recent successes) ──\n");
        let mut budget = max_chars.saturating_sub(out.len());
        for ep in self.episodes.iter().take(max_episodes) {
            let q = ep.query.replace('\n', " ").trim().to_string();
            let a = ep.answer.replace('\n', " ").trim().to_string();
            let line = format!("// Q: {q}\n// A: {a}\n");
            if line.len() > budget {
                break;
            }
            out.push_str(&line);
            budget -= line.len();
        }
        out
    }

    /// Run a full evolve cycle: seed population, parallel evaluate, self-play train.
    /// Returns the best candidate found.
    pub fn evolve(&mut self, source: &str) -> Option<Candidate> {
        use parallel_eval::ParallelEvaluator;
        use self_play::SelfPlayTrainer;

        // Seed initial population from source if archive is empty
        if self.candidates.is_empty() {
            let mut trainer = SelfPlayTrainer::new(&self.archive_path, self_play::SelfPlayConfig::default());
            let result = trainer.run(source);
            self.generation = result.best_generation;
            // Load best from archive file
            self.load();
        }

        // Parallel evaluation of current population
        let evaluator = ParallelEvaluator::new(parallel_eval::EvalConfig {
            max_workers: rayon::current_num_threads(),
            eval_timeout_secs: 120,
            fitness_tasks: Vec::new(),
        });
        let results = evaluator.evaluate_all(&self.candidates);

        // Update fitness from results
        for result in &results {
            if let Some(candidate) = self.candidates_mut()
                .iter_mut()
                .find(|c| c.id == result.candidate_id)
            {
                candidate.fitness = result.fitness;
                candidate.fitness_components.task_performance = result.task_performance;
                candidate.fitness_components.speed = result.speed;
                candidate.fitness_components.vfe = result.vfe;
                candidate.fitness_components.novelty = result.novelty;
                candidate.fitness = result.fitness;
            }
        }

        // Sort best first
        self.candidates_mut().sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });
        self.save();

        // Run self-play training to generate next generation
        let mut trainer = SelfPlayTrainer::new(&self.archive_path, self_play::SelfPlayConfig::default());
        let _result = trainer.run(source);

        // Reload best from archive
        self.load();
        self.candidates.first().cloned()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ArchiveData {
    candidates: Vec<Candidate>,
    generation: usize,
    threshold: f32,
    #[serde(default)]
    mutation_rate: f64,
    #[serde(default)]
    last_gen_best: f64,
    #[serde(default)]
    strict_promotions: u64,
    #[serde(default)]
    total_promotions: u64,
    #[serde(default)]
    min_promotion_delta: f64,
    #[serde(default)]
    episodes: Vec<Episode>,
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

/// Parallel evaluation pool — evaluates multiple candidates concurrently using rayon.
pub mod parallel_eval {
    use super::*;

    /// Configuration for parallel evaluation
    #[derive(Debug, Clone)]
    pub struct EvalConfig {
        pub max_workers: usize,
        pub eval_timeout_secs: u64,
        pub fitness_tasks: Vec<FitnessTask>,
    }

    /// A single fitness evaluation task for a candidate
    #[derive(Debug, Clone)]
    pub struct FitnessTask {
        pub candidate_id: String,
        pub patch: String,
        pub task_type: FitnessTaskType,
    }

    #[derive(Debug, Clone)]
    pub enum FitnessTaskType {
        Compile,
        TestSuite,
        VfeCheck,
        NoveltyCheck,
    }

    /// Result of evaluating a single candidate
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct EvalResult {
        pub candidate_id: String,
        pub fitness: f32,
        pub task_performance: f32,
        pub speed: f32,
        pub memory_efficiency: f32,
        pub vfe: f32,
        pub novelty: f32,
        pub success: bool,
    }

    /// Parallel evaluator — runs fitness evaluation across the thread pool
    pub struct ParallelEvaluator {
        pub config: EvalConfig,
    }

    impl ParallelEvaluator {
        pub fn new(config: EvalConfig) -> Self {
            Self { config }
        }

        /// Evaluate all candidates in parallel using rayon's thread pool.
        /// Each candidate's fitness is computed concurrently.
        pub fn evaluate_all(&self, candidates: &[Candidate]) -> Vec<EvalResult> {
            use rayon::prelude::*;

            candidates
                .par_iter()
                .map(|candidate| {
                    let start = std::time::Instant::now();

                    // Attempt compile
                    let compile_ok = self.try_compile(&candidate.patch);
                    let compile_time = start.elapsed().as_secs_f64();

                    // Speed bounded to [0,1]: `1/(1+t)` — instant ~1.0, 1s ~0.5.
                    // Previously `1/t` was unbounded (~1000 for instant compile),
                    // which inflated the composite to ~140 and made the fitness
                    // gate (threshold 0.8) trivially pass at generation 0 — the
                    // delta-gen pressure never actually bound.
                    let speed = (1.0 / (1.0 + compile_time)) as f32;

                    let mut result = EvalResult {
                        candidate_id: candidate.id.clone(),
                        success: compile_ok,
                        speed,
                        ..Default::default()
                    };

                    if compile_ok {
                        // Run test suite — a [0,1] pass rate.
                        if let Ok((pass_rate, _total)) = self.try_test_suite() {
                            result.task_performance = pass_rate;
                        }

                        // VFE proxy = test surprisal (1 - pass rate), clean and
                        // bounded; NOT derived from fitness (avoids circularity:
                        // fitness previously appeared on both sides).
                        result.vfe = (1.0 - result.task_performance).clamp(0.0, 1.0);

                        // Novelty: inverse of similarity to existing candidates
                        result.novelty = self.compute_novelty(&candidate.patch, candidates);

                        // Composite fitness — every component in [0,1], so a
                        // full-pass, fast, novel candidate tops out near ~0.9
                        // and the gate threshold (0.8) sits between "typical"
                        // and "elite" instead of below every score.
                        result.fitness = 0.4 * result.task_performance
                            + 0.2 * result.speed
                            + 0.2 * (1.0 - result.vfe)
                            + 0.2 * result.novelty;
                    }

                    result
                })
                .collect()
        }

        fn try_compile(&self, _patch: &str) -> bool {
            // Simplified: in production this would apply the patch, run cargo build, revert
            true
        }

        fn try_test_suite(&self) -> Result<(f32, usize), String> {
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

        fn compute_novelty(&self, _patch: &str, _population: &[Candidate]) -> f32 {
            // Simplified novelty metric — in production use structural diff or embedding distance
            0.5 + (rand::random::<f32>() * 0.5)
        }
    }
}

/// Self-play training loop — generates, evaluates, and evolves a population of patches.
pub mod self_play {
    use super::*;
    use rand::seq::SliceRandom;

    /// Configuration for the self-play training loop
    #[derive(Debug, Clone, Copy)]
    pub struct SelfPlayConfig {
        pub population_size: usize,
        pub max_generations: usize,
        pub mutation_rate: f32,
        pub crossover_rate: f32,
        pub fitness_threshold: f32,
        pub elite_count: usize,
        pub temperature: f32,
    }

    impl Default for SelfPlayConfig {
        fn default() -> Self {
            Self {
                population_size: 20,
                max_generations: 50,
                mutation_rate: 0.15,
                crossover_rate: 0.3,
                fitness_threshold: 0.8,
                elite_count: 3,
                temperature: 1.0,
            }
        }
    }

    /// Training result from self-play
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SelfPlayResult {
        pub best_fitness: f32,
        pub best_generation: usize,
        pub final_population_size: usize,
        pub convergence_generations: usize,
        pub fitness_history: Vec<f32>,
        pub best_patch: String,
    }

    /// Self-play trainer — evolves a population of code patches through
    /// genetic operators and parallel fitness evaluation.
    pub struct SelfPlayTrainer {
        pub config: SelfPlayConfig,
        pub archive: DarwinArchive,
    }

    impl SelfPlayTrainer {
        pub fn new(archive_path: &str, config: SelfPlayConfig) -> Self {
            Self {
                config,
                archive: DarwinArchive::new(archive_path, config.fitness_threshold, config.population_size),
            }
        }

        /// Run the self-play training loop.
        /// Returns the best fitness score and patch found.
        pub fn run(&mut self, initial_source: &str) -> SelfPlayResult {
            let evaluator = parallel_eval::ParallelEvaluator::new(parallel_eval::EvalConfig {
                max_workers: rayon::current_num_threads(),
                eval_timeout_secs: 120,
                fitness_tasks: Vec::new(),
            });

            let mut fitness_history = Vec::new();
            let mut converged_at = self.config.max_generations;
            // Patience counter: how many consecutive generations failed to
            // clear the delta-gen gate. Convergence is NOT an absolute fitness
            // ceiling (every candidate passes the current suite, so `>= thr`
            // fires at generation 0 and the gate never binds) — it is plateau
            // detection: stop when nothing has strictly improved for N gens.
            let mut no_improve_streak = 0usize;
            const CONVERGE_PATIENCE: usize = 5;

            // Seed initial population from the base source
            self.seed_population(initial_source);

            for generation in 0..self.config.max_generations {
                // Evaluate all candidates in parallel
                let results = evaluator.evaluate_all(self.archive.candidates());

                // Update fitness from eval results
                for result in &results {
                    if let Some(candidate) = self.archive.candidates_mut()
                        .iter_mut()
                        .find(|c| c.id == result.candidate_id)
                    {
                        candidate.fitness = result.fitness;
                        candidate.fitness_components.task_performance = result.task_performance;
                        candidate.fitness_components.speed = result.speed;
                        candidate.fitness_components.vfe = result.vfe;
                        candidate.fitness_components.novelty = result.novelty;
                    }
                }

                // Sort by fitness (descending)
                self.archive.candidates_mut().sort_by(|a, b| {
                    b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
                });

                let best_fitness = self.archive.candidates.first().map(|c| c.fitness).unwrap_or(0.0);
                fitness_history.push(best_fitness);

                // Safety-gated champion election (LAYER 3b wired into
                // self-play): promote the best candidate only if it strictly
                // improves over the last gated baseline AND passes the darwin
                // safety invariants. `promote_best_gated` returns None when
                // nothing cleared the gate — the goalposts never move
                // backwards, and an invariant-breaking best candidate is
                // refused rather than promoted.
                let promoted = self.archive.promote_best_gated();
                let champion_fitness = promoted.as_ref().map(|c| c.fitness).unwrap_or(best_fitness);

                // Mutation-rate adaptation stays driven by the observed delta
                // (narrow on improvement, widen on regression/plateau) no
                // matter whether a champion cleared the safety gate.
                let delta = champion_fitness as f64 - self.archive.last_gen_fitness();
                self.archive.adapt_mutation_rate(delta);

                // Patience-based convergence: strict improvement resets the
                // streak; CONVERGE_PATIENCE plateaus in a row stop the run.
                if promoted.is_some() {
                    no_improve_streak = 0;
                } else {
                    no_improve_streak += 1;
                }
                if no_improve_streak >= CONVERGE_PATIENCE {
                    converged_at = generation;
                    break;
                }

                // Generate next generation
                self.next_generation();
            }

            // Persist the evolved state (candidates, generation, fitness gate,
            // mutation rate, episodes) so evolution is continuous across nightly
            // runs — the archive must survive this process exiting.
            self.archive.save();

            let best = self.archive.candidates.first().cloned();
            SelfPlayResult {
                best_fitness: best.as_ref().map(|c| c.fitness).unwrap_or(0.0),
                best_generation: if fitness_history.is_empty() { 0 } else { fitness_history.len() - 1 },
                final_population_size: self.archive.candidates.len(),
                convergence_generations: converged_at,
                fitness_history,
                best_patch: best.as_ref().map(|c| c.patch.clone()).unwrap_or_default(),
            }
        }

        fn seed_population(&mut self, initial_source: &str) {
            // LAYER 3e: fold the episodic replay digest into the mutation
            // context as a comment block. Syn parses it (comments are valid
            // Rust), so mutations operate on code that carries the trace of
            // recent successful interactions. If no episodes exist yet this
            // is a no-op and behavior is unchanged.
            let replay_ctx = self.archive.replay_digest(16, 6000);
            let seed_source = if self.archive.episode_count() > 0 && !replay_ctx.is_empty() {
                format!("{replay_ctx}\n{initial_source}")
            } else {
                initial_source.to_string()
            };

            // Generate initial mutations from the base source
            for i in 0..self.config.population_size {
                if i == 0 {
                    // First candidate is the baseline (identity)
                    self.archive.candidates.push(Candidate {
                        id: format!("baseline_{}", i),
                        description: "Baseline: no mutation".to_string(),
                        patch: String::new(),
                        fitness: 0.0,
                        fitness_components: FitnessComponents::default(),
                        generation: 0,
                        parent_id: None,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    });
                } else {
                    // Mutate the source to create a new candidate
                    if let Ok(mutation) = ast_mutator::random_mutation(&seed_source) {
                        if let Ok(patch) = ast_mutator::apply_mutation(&seed_source, &mutation) {
                            let parent = self.archive.candidates.first().map(|c| c.id.clone());
                            self.archive.candidates.push(Candidate {
                                id: format!("seed_{}", i),
                                description: format!("Seed mutation {}", i),
                                patch,
                                fitness: 0.0,
                                fitness_components: FitnessComponents::default(),
                                generation: 0,
                                parent_id: parent,
                                timestamp: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                            });
                        }
                    }
                }
            }
        }

        fn next_generation(&mut self) {
            let mut rng = rand::thread_rng();
            let current = std::mem::take(&mut self.archive.candidates);
            let elite_count = self.config.elite_count.min(current.len());

            // Keep elites
            let elites: Vec<Candidate> = current.iter().take(elite_count).cloned().collect();
            let remaining = self.config.population_size - elite_count;

            let mut new_candidates = elites;

            for _ in 0..remaining {
                let parent_a = current.choose(&mut rng).unwrap();
                let parent_b = current.choose(&mut rng).unwrap();

                if rng.gen::<f32>() < self.config.crossover_rate {
                    // Crossover
                    let child_patch = genetic::crossover(&parent_a.patch, &parent_b.patch);
                    let child = Candidate {
                        id: format!("cand_{}", rng.gen::<u32>()),
                        description: format!("Crossover of {} and {}", parent_a.id, parent_b.id),
                        patch: child_patch,
                        fitness: 0.0,
                        fitness_components: FitnessComponents::default(),
                        generation: self.archive.generation + 1,
                        parent_id: Some(parent_a.id.clone()),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    };
                    new_candidates.push(child);
                } else {
                    // Mutation — depth (`rounds`) is scaled by the adaptive
                    // mutation rate: wide rate (exploration) applies more
                    // mutations per child, narrow rate (exploitation) fewer.
                    let rounds = (1 + self.archive.mutation_rate().round() as usize).clamp(1, 3);
                    let mut patch = parent_a.patch.clone();
                    let mut ok = true;
                    for _ in 0..rounds {
                        if let Ok(mutation) = ast_mutator::random_mutation(&patch) {
                            match ast_mutator::apply_mutation(&patch, &mutation) {
                                Ok(p) => patch = p,
                                Err(_) => { ok = false; break; }
                            }
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        let child = Candidate {
                            id: format!("cand_{}", rng.gen::<u32>()),
                            description: format!("Mutation of {}", parent_a.id),
                            patch,
                            fitness: 0.0,
                            fitness_components: FitnessComponents::default(),
                            generation: self.archive.generation + 1,
                            parent_id: Some(parent_a.id.clone()),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        };
                        new_candidates.push(child);
                    }
                }
            }

            self.archive.candidates = new_candidates;
            self.archive.generation += 1;
        }
    }
}
// ── Episodic replay: ingest from bridge chat memory shards ───────────────
//
// The bridge absorbs every (query → answer) conversation into
// `.kai_chat_memory.json` (JSON: {"entries": {"__chat__/<ts>": {"text":
// "Q: …\nA: …", "ts": …, "kind": "chat"}}}). `load_chat_episodes` parses any
// number of matching shard files and returns the episodes newest-first.

/// Parse a bridge-style chat-memory JSON file (v2 format) into episodes.
/// Returns an error string if the file cannot be parsed (file missing is Ok
/// and yields an empty vec so callers can glob optional shards).
pub fn parse_chat_memory(path: &str) -> Result<Vec<Episode>, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| format!("{path}: {e}"))?;
    let entries = v
        .get("entries")
        .and_then(|e| e.as_object())
        .ok_or_else(|| format!("{path}: no entries object"))?;
    let mut out = Vec::new();
    for (key, entry) in entries {
        let text = entry.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let ts = entry.get("ts").and_then(|t| t.as_u64()).unwrap_or(0);
        let kind = entry.get("kind").and_then(|k| k.as_str()).unwrap_or("chat").to_string();
        // text is "Q: …\nA: …" — split into query/answer.
        let (q, a) = match text.split_once("\nA: ") {
            Some((qpart, apart)) => {
                let q = qpart.trim_start_matches("Q: ").trim().to_string();
                (q, apart.trim().to_string())
            }
            None => (text.trim().to_string(), String::new()),
        };
        if q.is_empty() && a.is_empty() {
            continue;
        }
        out.push(Episode { query: q, answer: a, ts, kind });
        let _ = key;
    }
    // Newest first
    out.sort_by(|a, b| b.ts.cmp(&a.ts));
    Ok(out)
}

/// Glob `pattern` (may contain `*`) and parse every matching chat-memory
/// shard, merging episodes newest-first. Missing/empty matches return empty.
pub fn load_chat_episodes(pattern: &str) -> Vec<Episode> {
    let (dir, glob_part) = match pattern.rfind('/') {
        Some(idx) => (&pattern[..idx], &pattern[idx + 1..]),
        None => (".", pattern),
    };
    let mut episodes = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !glob_matches(&name, glob_part) {
                continue;
            }
            if let Ok(mut eps) = parse_chat_memory(&entry.path().to_string_lossy()) {
                episodes.append(&mut eps);
            }
        }
    }
    episodes.sort_by(|a, b| b.ts.cmp(&a.ts));
    episodes
}

/// Minimal glob: `*` matches any run of characters (used for shard names).
fn glob_matches(name: &str, glob: &str) -> bool {
    if !glob.contains('*') {
        return name == glob;
    }
    let parts: Vec<&str> = glob.split('*').collect();
    if parts.is_empty() {
        return true;
    }
    // start..end anchored; middle must appear in order
    let mut rest = name;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match rest.find(part) {
            Some(pos) => {
                if i == 0 && !rest.starts_with(part) {
                    return false;
                }
                rest = &rest[pos + part.len()..];
            }
            None => return false,
        }
    }
    // Last part must be a suffix if it was non-empty
    if let Some(last) = parts.last() {
        if !last.is_empty() && !name.ends_with(last) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_candidate(id: &str, fitness: f32) -> Candidate {
        Candidate {
            id: id.to_string(),
            description: format!("test candidate {id}"),
            patch: format!("fn {id}() {{}}"),
            fitness,
            fitness_components: FitnessComponents::default(),
            generation: 0,
            parent_id: None,
            timestamp: 0,
        }
    }

    #[test]
    fn test_promote_gated_refuses_flat_regression() {
        // baseline 0.9; only a 0.85 candidate exists -> no strict improvement.
        let mut a = DarwinArchive::new("/tmp/kai_darwin_gate_test_flat.json", 0.5, 20);
        a.last_gen_best = 0.9;
        a.candidates.push(mk_candidate("c1", 0.85));
        assert!(a.promote_best_gated().is_none(), "regression must be refused");
        assert_eq!(a.total_promotions, 1);
        assert_eq!(a.strict_promotions, 0);
    }

    #[test]
    fn test_promote_gated_allows_improvement() {
        // baseline 0.9; a candidate with a meaningful improvement (delta >
        // min_promotion_delta) must promote. We use 0.97 so the fitness delta
        // (0.07) clearly exceeds the tightened promotion floor (0.05) and side
        // steps f32->f64 representation loss at the boundary.
        let mut a = DarwinArchive::new("/tmp/kai_darwin_gate_test_impr.json", 0.5, 20);
        a.last_gen_best = 0.9;
        a.candidates.push(mk_candidate("c1", 0.97));
        let p = a.promote_best_gated().expect("strict improvement must promote");
        assert_eq!(p.id, "c1");
        assert_eq!(a.strict_promotions, 1);
        // baseline advances to the promoted fitness
        assert!((a.last_gen_best - 0.97).abs() < 1e-6);
    }

    #[test]
    fn test_promote_gated_honors_safety() {
        // Archive with one "unsafe" candidate; gated promotion must refuse
        // rather than promote an invariant-breaking patch.
        let mut a = DarwinArchive::new("/tmp/kai_darwin_gate_test_unsafe.json", 0.5, 20);
        a.last_gen_best = 0.9;
        // A patch that touches the destructive-filesystem-operation pattern
        // (std::fs::remove*) and therefore must be refused by the safety gate.
        let mut c = mk_candidate("unsafe", 0.99);
        c.patch = "fn unsafe_run() { std::fs::remove_file(\"/tmp/x\"); }".to_string();
        a.candidates.push(c);
        let p = a.promote_best_gated();
        assert!(p.is_none(), "unsafe candidate must not be promoted");
        assert_eq!(a.strict_promotions, 0);
        let _ = std::fs::remove_file("/tmp/kai_darwin_gate_test_flat.json");
        let _ = std::fs::remove_file("/tmp/kai_darwin_gate_test_impr.json");
        let _ = std::fs::remove_file("/tmp/kai_darwin_gate_test_unsafe.json");
    }

    #[test]
    fn test_self_play_gate_refuses_safety_breaking_champion() {
        // The self-play trainer elects its champion through promote_best_gated
        // (the same gate run() uses). A safety-breaking candidate that would
        // win on fitness alone must be refused — the trainer never promotes an
        // invariant-breaking patch even when it tops the population.
        let mut trainer = self_play::SelfPlayTrainer::new(
            "/tmp/kai_selfplay_gate_test.json",
            self_play::SelfPlayConfig::default(),
        );
        trainer.archive.last_gen_best = 0.5;
        let mut unsafe_best = mk_candidate("unsafe", 0.99);
        unsafe_best.patch = "fn unsafe_run() { std::fs::remove_file(\"/tmp/x\"); }".to_string();
        trainer.archive.candidates.push(unsafe_best);
        trainer.archive.candidates.push(mk_candidate("safe_low", 0.6));
        // Sort best-first exactly as the trainer loop does before electing.
        trainer.archive.candidates.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });
        // baseline = max(last_gen_best, threshold) = 0.8; the unsafe 0.99 must
        // be skipped by the safety gate, and 0.6 is not a strict improvement.
        let promoted = trainer.archive.promote_best_gated();
        assert!(promoted.is_none(), "trainer gate must refuse unsafe best");
        assert_eq!(trainer.archive.strict_promotions, 0);
        let _ = std::fs::remove_file("/tmp/kai_selfplay_gate_test.json");
    }

    #[test]
    fn test_adapt_mutation_rate_tightens_on_improvement() {
        let mut a = DarwinArchive::new("/tmp/kai_darwin_rate_test.json", 0.5, 20);
        assert!((a.mutation_rate() - 1.0).abs() < 1e-9);
        a.adapt_mutation_rate(0.05); // improvement -> shrink step
        assert!(a.mutation_rate() < 1.0, "improvement should shrink rate");
        a.adapt_mutation_rate(-0.05); // regression -> widen
        assert!(a.mutation_rate() > 0.9, "regression should widen rate");
        let _ = std::fs::remove_file("/tmp/kai_darwin_rate_test.json");
    }

    #[test]
    fn test_strict_improvement_ratio() {
        let mut a = DarwinArchive::new("/tmp/kai_darwin_ratio_test.json", 0.5, 20);
        assert_eq!(a.strict_improvement_ratio(), 0.0);
        a.total_promotions = 4;
        a.strict_promotions = 3;
        assert!((a.strict_improvement_ratio() - 0.75).abs() < 1e-6);
        let _ = std::fs::remove_file("/tmp/kai_darwin_ratio_test.json");
    }

    // ── Episodic replay tests ──────────────────────────────────────────

    #[test]
    fn test_parse_chat_memory_v2() {
        // Emulate a bridge v2 chat shard on disk.
        let path = "/tmp/kai_darwin_chat_test.json";
        std::fs::write(path, r#"{"version":2,"total_entries":2,"entries":{
            "__chat__/100": {"text":"Q: What is a kernel?\nA: It manages processes.","ts":100,"kind":"chat"},
            "__chat__/200": {"text":"Q: Define curvature.\nA: g_ij novelty-driven metric term.","ts":200,"kind":"chat"}
        }}"#).unwrap();
        let eps = parse_chat_memory(path).unwrap();
        assert_eq!(eps.len(), 2);
        // newest first
        assert_eq!(eps[0].query, "Define curvature.");
        assert_eq!(eps[0].answer, "g_ij novelty-driven metric term.");
        assert_eq!(eps[0].ts, 200);
        assert_eq!(eps[1].query, "What is a kernel?");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_episode_ingest_caps_and_dedups() {
        let mut a = DarwinArchive::new("/tmp/kai_darwin_ep_test.json", 0.5, 20);
        assert_eq!(a.episode_count(), 0);
        // Identical episode twice -> dedup keeps one.
        let ep = Episode { query: "q".into(), answer: "a deliberately long enough answer body".into(), ts: 1, kind: "chat".into() };
        assert!(a.ingest_episode(ep.clone()));
        assert!(!a.ingest_episode(ep));
        assert_eq!(a.episode_count(), 1);
        // Newer episode arrives -> 2 total, newest first in digest.
        let ep2 = Episode { query: "q2".into(), answer: "another long enough answer body here".into(), ts: 9, kind: "chat".into() };
        assert!(a.ingest_episode(ep2));
        assert_eq!(a.episode_count(), 2);
        let digest = a.replay_digest(10, 2000);
        assert!(digest.contains("q2"), "newest must appear first in digest");
        // Reject a too-thin answer (not a successful episode).
        let thin = Episode { query: "q3".into(), answer: "ok".into(), ts: 10, kind: "chat".into() };
        assert!(!a.ingest_episode(thin));
        let _ = std::fs::remove_file("/tmp/kai_darwin_ep_test.json");
    }

    #[test]
    fn test_load_chat_episodes_glob() {
        // Two shard files; glob merge must read both and sort newest-first.
        std::fs::write("/tmp/kai_darwin_shard_0.json", r#"{"entries":{"a":{"text":"Q: one\nA: first answer body","ts":1,"kind":"chat"}}}"#).unwrap();
        std::fs::write("/tmp/kai_darwin_shard_1.json", r#"{"entries":{"b":{"text":"Q: two\nA: second answer body","ts":2,"kind":"chat"}}}"#).unwrap();
        let eps = load_chat_episodes("/tmp/kai_darwin_shard_*.json");
        assert_eq!(eps.len(), 2);
        assert_eq!(eps[0].query, "two");
        assert_eq!(eps[1].query, "one");
        let _ = std::fs::remove_file("/tmp/kai_darwin_shard_0.json");
        let _ = std::fs::remove_file("/tmp/kai_darwin_shard_1.json");
    }

    #[test]
    fn test_glob_matches_helper() {
        assert!(glob_matches(".kai_chat_memory.json", "*.json"));
        assert!(glob_matches(".kai_chat_memory.0.json", "*.kai_chat_memory*.json"));
        assert!(!glob_matches("other.txt", "*.json"));
        assert!(glob_matches("exact", "exact"));
    }
}
