//! Multi-agent routing: classifies queries and dispatches to specialized sub-agents.
//! Each sub-agent has a different system prompt tuned for its domain.
//! All agents share the same attractor (for fixed-point convergence).
//!
//! Phase 6.3: Multi-Agent Architecture
//! ```
//!          ┌─────────────────────────┐
//!          │    Metacognitive Agent   │
//!          │  routes.rs: classify()  │
//!          └────────┬────────────────┘
//!                   │
//!     ┌─────────────┼─────────────┐
//!     │             │             │
//!     ▼             ▼             ▼
//! ┌────────┐ ┌────────────┐ ┌──────────┐
//! │ Code   │ │ Math/Sci   │ │ Creative │
//! │ Agent  │ │ Agent      │ │ Agent    │
//! └────────┘ └────────────┘ └──────────┘
//!     │             │             │
//!     └─────────────┼─────────────┘
//!                   │
//!          ┌────────▼────────┐
//!          │   Shared Memory  │
//!          │  (attractor.rs)  │
//!          └─────────────────┘
//! ```

/// Agent kinds for specialized sub-agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    /// Code generation, debugging, refactoring
    Code,
    /// Mathematics, science, logic, analysis
    MathScience,
    /// Creative writing, brainstorming, design
    Creative,
    /// Summarization, research, information extraction
    Analysis,
    /// Default general-purpose agent
    General,
}

impl AgentKind {
    /// All agent kinds for iteration.
    #[allow(dead_code)]
    pub fn all() -> Vec<AgentKind> {
        vec![
            AgentKind::Code,
            AgentKind::MathScience,
            AgentKind::Creative,
            AgentKind::Analysis,
            AgentKind::General,
        ]
    }

    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            AgentKind::Code => "Code",
            AgentKind::MathScience => "Math/Science",
            AgentKind::Creative => "Creative",
            AgentKind::Analysis => "Analysis",
            AgentKind::General => "General",
        }
    }

    /// Emoji icon for display.
    pub fn icon(&self) -> &str {
        match self {
            AgentKind::Code => "💻",
            AgentKind::MathScience => "🔬",
            AgentKind::Creative => "🎨",
            AgentKind::Analysis => "📊",
            AgentKind::General => "🤖",
        }
    }
}

/// Classify a user query into an agent kind based on keyword analysis.
/// Simple keyword-based router — could be upgraded to an LLM-based classifier.
pub fn classify_query(query: &str) -> AgentKind {
    let lower = query.to_lowercase();

    // Helper: check for whole-word match (word boundaries or start/end of string)
    let contains_word = |s: &str, word: &str| -> bool {
        if word.is_empty() { return false; }
        // Check each occurrence of word in s using char-boundary word check
        let mut start = 0;
        while let Some(pos) = s[start..].find(word) {
            let abs_pos = start + pos;
            let before = abs_pos == 0 || !s.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
            let after = abs_pos + word.len() >= s.len() || !s.as_bytes()[abs_pos + word.len()].is_ascii_alphanumeric();
            if before && after {
                return true;
            }
            start = abs_pos + 1;
        }
        false
    };

    // Code signals
    let code_keywords = [
        "code", "function", "class", "impl", "rust", "python",
        "javascript", "typescript", "debug", "compile", "error", "bug",
        "refactor", "algorithm", "implementation", "api", "endpoint",
        "git", "commit", "merge", "syntax", "type", "variable",
        "loop", "array", "string", "import", "module", "crate",
        "struct", "enum",
    ];
    for kw in &code_keywords {
        if contains_word(&lower, kw) {
            return AgentKind::Code;
        }
    }
    // Also check for code-specific patterns
    if lower.contains("fn ") || lower.contains("->") || lower.contains("fn(") || lower.contains("mut ") || lower.contains("let ") || lower.contains("pub ") {
        return AgentKind::Code;
    }

    // Math/Science signals
    let math_keywords = [
        "equation", "formula", "calculate", "compute", "derivative",
        "integral", "probability", "statistics", "variance", "physics",
        "quantum", "relativity", "thermodynamics", "chemistry",
        "biology", "hypothesis", "experiment", "proof",
        "theorem", "lemma", "axiom",
    ];
    for kw in &math_keywords {
        if contains_word(&lower, kw) {
            return AgentKind::MathScience;
        }
    }
    // Also check for math-specific patterns
    if lower.contains("x^") || lower.contains(" solve ") || lower.contains("calculate") {
        return AgentKind::MathScience;
    }

    // Creative signals
    let creative_keywords = [
        "poem", "poetry", "creative", "design", "art", "music",
        "brainstorm", "imagine", "narrative", "metaphor",
        "invent", "aesthetic",
    ];
    for kw in &creative_keywords {
        if contains_word(&lower, kw) {
            return AgentKind::Creative;
        }
    }
    // Also check for creative-specific looser patterns
    if lower.contains("write a ") || lower.contains("write an ") || lower.contains("generate a ") || lower.contains("write me ") {
        return AgentKind::Creative;
    }

    // Analysis signals
    let analysis_keywords = [
        "analysis", "analyze", "summarize", "summarise", "research",
        "compare", "contrast", "evaluate", "assess", "review",
        "investigate", "examine", "findings",
    ];
    for kw in &analysis_keywords {
        if contains_word(&lower, kw) {
            return AgentKind::Analysis;
        }
    }
    if lower.contains("tl;dr") || lower.contains("key points") || lower.contains("break down") {
        return AgentKind::Analysis;
    }

    // Default to general
    AgentKind::General
}

/// Get the system prompt for a given agent kind.
/// These prompts tune the model's behavior for its specialization.
pub fn system_prompt(kind: AgentKind) -> &'static str {
    match kind {
        AgentKind::Code => "You are a senior software engineer. \
            Write clear, idiomatic, well-commented code. \
            Prefer Rust unless otherwise specified. \
            Consider edge cases, error handling, and performance. \
            Always explain your reasoning briefly before code.",

        AgentKind::MathScience => "You are a mathematician and scientist. \
            Reason step-by-step using first principles. \
            Show your work and verify each step. \
            Use precise mathematical notation. \
            If uncertain, state your confidence level explicitly. \
            Prefer rigorous proofs over intuition.",

        AgentKind::Creative => "You are a creative artist and writer. \
            Think outside the box. Use vivid language and imagery. \
            Explore multiple perspectives. \
            Prioritize originality and emotional resonance. \
            Don't be afraid to be unconventional.",

        AgentKind::Analysis => "You are a research analyst. \
            Be thorough, objective, and evidence-based. \
            Structure your analysis: overview → details → conclusion. \
            Identify key patterns, relationships, and implications. \
            Cite sources and note uncertainty. \
            Synthesize information from multiple angles.",

        AgentKind::General => "You are Kai-Fusion, a general-purpose AI assistant \
            powered by a physics-wired transformer with variational free energy \
            minimization. You are helpful, honest, and precise.",
    }
}

/// Route configuration for a multi-agent dispatch.
#[derive(Debug, Clone)]
pub struct RouteConfig {
    /// Which sub-agent to route to
    pub kind: AgentKind,
    /// System prompt for this agent
    pub system_prompt: &'static str,
    /// Temperature override (None = use default)
    pub temperature: Option<f32>,
    /// Top-p override
    pub top_p: Option<f32>,
}

impl RouteConfig {
    /// Create a RouteConfig from a query using automatic classification.
    pub fn from_query(query: &str) -> Self {
        let kind = classify_query(query);
        RouteConfig {
            kind,
            system_prompt: system_prompt(kind),
            temperature: None,
            top_p: None,
        }
    }

    /// Create a RouteConfig with explicit agent kind.
    pub fn from_kind(kind: AgentKind) -> Self {
        RouteConfig {
            kind,
            system_prompt: system_prompt(kind),
            temperature: None,
            top_p: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_code_query() {
        assert_eq!(classify_query("write a rust function to sort an array"), AgentKind::Code);
        assert_eq!(classify_query("fix this bug in my python code"), AgentKind::Code);
        assert_eq!(classify_query("how do I use the map function in javascript"), AgentKind::Code);
        assert_eq!(classify_query("implement an api endpoint in rust"), AgentKind::Code);
    }

    #[test]
    fn test_classify_math_query() {
        assert_eq!(classify_query("solve the equation x^2 + 5x + 6 = 0"), AgentKind::MathScience);
        assert_eq!(classify_query("explain the theory of relativity"), AgentKind::MathScience);
        assert_eq!(classify_query("calculate probability of rolling a 6"), AgentKind::MathScience);
        assert_eq!(classify_query("prove the quadratic formula"), AgentKind::MathScience);
    }

    #[test]
    fn test_classify_creative_query() {
        assert_eq!(classify_query("write a poem about AI"), AgentKind::Creative);
        assert_eq!(classify_query("brainstorm ideas for a startup"), AgentKind::Creative);
        assert_eq!(classify_query("design a logo for my brand"), AgentKind::Creative);
    }

    #[test]
    fn test_classify_analysis_query() {
        assert_eq!(classify_query("summarize this article"), AgentKind::Analysis);
        assert_eq!(classify_query("analyze the pros and cons"), AgentKind::Analysis);
        assert_eq!(classify_query("compare these two approaches"), AgentKind::Analysis);
    }

    #[test]
    fn test_classify_general_query() {
        assert_eq!(classify_query("hello how are you"), AgentKind::General);
        assert_eq!(classify_query("what is the weather like"), AgentKind::General);
        assert_eq!(classify_query("tell me about yourself"), AgentKind::General);
    }

    #[test]
    fn test_route_config_from_query() {
        let route = RouteConfig::from_query("write a rust function");
        assert_eq!(route.kind, AgentKind::Code);
        assert!(route.system_prompt.contains("software engineer"));
    }

    #[test]
    fn test_agent_kind_labels() {
        for kind in AgentKind::all() {
            assert!(!kind.label().is_empty());
            assert!(!kind.icon().is_empty());
        }
    }

    #[test]
    fn test_system_prompts_all_unique() {
        let prompts: std::collections::HashSet<&str> = AgentKind::all()
            .iter()
            .map(|k| system_prompt(*k))
            .collect();
        assert_eq!(prompts.len(), AgentKind::all().len(),
            "each agent kind should have a unique system prompt");
    }
}
