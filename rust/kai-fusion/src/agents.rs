//! Phase 6.3: Multi-Agent Architecture — specialized sub-agents with shared memory.
//!
//! From AGI_PLAN.md §6.3:
//! "Each sub-agent runs on the same LLM with different system prompts
//! and tool sets. The metacognitive agent routes tasks."
//!
//! Architecture:
//!   Metacognitive Agent (router/planner)
//!     ├── Code Agent (write/test/debug code)
//!     ├── Math/Science Agent (reasoning, calculation)
//!     └── Creative Agent (brainstorm, generate ideas)
//!   All share: SharedMemory (SQLite + vector DB)

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Agent role — specialization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// Metacognitive: plans, monitors, routes tasks
    MetaCognitive,
    /// Code: writes, tests, debugs code
    Code,
    /// Math/Science: reasoning, calculation, analysis
    MathSci,
    /// Creative: brainstorming, generation, ideation
    Creative,
    /// Generic fallback
    Generic,
}

impl AgentRole {
    pub fn label(&self) -> &'static str {
        match self {
            AgentRole::MetaCognitive => "metacognitive",
            AgentRole::Code => "code",
            AgentRole::MathSci => "math/science",
            AgentRole::Creative => "creative",
            AgentRole::Generic => "generic",
        }
    }

    pub fn system_prompt(&self) -> &'static str {
        match self {
            AgentRole::MetaCognitive =>
                "You are the metacognitive agent. You plan, monitor, set goals, resolve conflicts, and route tasks to specialized agents.",
            AgentRole::Code =>
                "You are the code agent. You write, test, debug, and optimize code. Focus on correctness, performance, and maintainability.",
            AgentRole::MathSci =>
                "You are the math/science agent. You perform reasoning, calculation, analysis, and scientific investigation.",
            AgentRole::Creative =>
                "You are the creative agent. You brainstorm, generate novel ideas, create content, and think outside the box.",
            AgentRole::Generic =>
                "You are a helpful assistant.",
        }
    }

    pub fn all() -> &'static [AgentRole] {
        &[
            AgentRole::MetaCognitive,
            AgentRole::Code,
            AgentRole::MathSci,
            AgentRole::Creative,
        ]
    }
}

/// Task to be routed to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Task identifier
    pub id: u64,
    /// Task description/prompt
    pub prompt: String,
    /// Assigned role (or None if not yet routed)
    pub assigned_role: Option<AgentRole>,
    /// Priority (higher = more urgent)
    pub priority: u8,
    /// Dependencies (task IDs that must complete first)
    pub dependencies: Vec<u64>,
    /// Result (if completed)
    pub result: Option<String>,
    /// Status
    pub status: TaskStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Routed,
    InProgress,
    Completed,
    Failed { reason: String },
}

/// Shared memory between agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemory {
    /// Key-value store for general state
    pub kv: HashMap<String, String>,
    /// Conversation history per agent role
    pub history: HashMap<String, Vec<(String, String)>>,
    /// Artifacts produced by agents
    pub artifacts: HashMap<String, String>,
}

impl SharedMemory {
    pub fn new() -> Self {
        SharedMemory {
            kv: HashMap::new(),
            history: HashMap::new(),
            artifacts: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.kv.insert(key.to_string(), value.to_string());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.kv.get(key).map(|s| s.as_str())
    }

    pub fn push_history(&mut self, agent: &str, role: &str, message: &str) {
        self.history
            .entry(agent.to_string())
            .or_default()
            .push((role.to_string(), message.to_string()));
    }

    pub fn store_artifact(&mut self, name: &str, content: &str) {
        self.artifacts.insert(name.to_string(), content.to_string());
    }

    pub fn history_len(&self, agent: &str) -> usize {
        self.history.get(agent).map(|h| h.len()).unwrap_or(0)
    }
}

impl Default for SharedMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Router — decides which agent handles a task based on keywords.
pub struct TaskRouter;

impl TaskRouter {
    /// Route a task to the best agent based on keyword analysis.
    pub fn route(task: &mut Task) {
        let lower = task.prompt.to_lowercase();

        let role = if lower.contains("code") || lower.contains("function")
            || lower.contains("implement") || lower.contains("debug")
            || lower.contains("rust") || lower.contains("python")
            || lower.contains("compile") || lower.contains("test")
        {
            AgentRole::Code
        } else if lower.contains("calculate") || lower.contains("math")
            || lower.contains("proof") || lower.contains("equation")
            || lower.contains("analyze") || lower.contains("statistics")
            || lower.contains("science") || lower.contains("physics")
        {
            AgentRole::MathSci
        } else if lower.contains("create") || lower.contains("brainstorm")
            || lower.contains("idea") || lower.contains("design")
            || lower.contains("story") || lower.contains("creative")
            || lower.contains("imagine") || lower.contains("write")
        {
            AgentRole::Creative
        } else if lower.contains("plan") || lower.contains("route")
            || lower.contains("monitor") || lower.contains("goal")
            || lower.contains("priority") || lower.contains("assess")
        {
            AgentRole::MetaCognitive
        } else {
            // Default to metacognitive for generic tasks
            AgentRole::MetaCognitive
        };

        task.assigned_role = Some(role);
        task.status = TaskStatus::Routed;
    }
}

/// Multi-Agent Manager — orchestrates agents, routes tasks, manages shared state.
pub struct MultiAgentManager {
    pub shared_memory: SharedMemory,
    tasks: Vec<Task>,
    next_task_id: u64,
    /// Per-role task count
    role_counts: HashMap<AgentRole, u32>,
}

impl MultiAgentManager {
    pub fn new() -> Self {
        MultiAgentManager {
            shared_memory: SharedMemory::new(),
            tasks: Vec::new(),
            next_task_id: 1,
            role_counts: HashMap::new(),
        }
    }

    /// Submit a new task. Returns the task ID.
    pub fn submit_task(&mut self, prompt: &str, priority: u8) -> u64 {
        let id = self.next_task_id;
        self.next_task_id += 1;
        let mut task = Task {
            id,
            prompt: prompt.to_string(),
            assigned_role: None,
            priority,
            dependencies: Vec::new(),
            result: None,
            status: TaskStatus::Pending,
        };
        TaskRouter::route(&mut task);
        if let Some(role) = task.assigned_role {
            *self.role_counts.entry(role).or_insert(0) += 1;
        }
        self.tasks.push(task);
        id
    }

    /// Get a task by ID.
    pub fn get_task(&self, id: u64) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == id)
    }

    /// Get all tasks for a given role.
    pub fn tasks_for_role(&self, role: AgentRole) -> Vec<&Task> {
        self.tasks.iter()
            .filter(|t| t.assigned_role == Some(role))
            .collect()
    }

    /// Get pending tasks (sorted by priority, descending).
    pub fn pending_tasks(&self) -> Vec<&Task> {
        let mut pending: Vec<_> = self.tasks.iter()
            .filter(|t| t.status == TaskStatus::Pending || t.status == TaskStatus::Routed)
            .collect();
        pending.sort_by(|a, b| b.priority.cmp(&a.priority));
        pending
    }

    /// Complete a task with a result.
    pub fn complete_task(&mut self, id: u64, result: &str) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.result = Some(result.to_string());
            task.status = TaskStatus::Completed;
            // Store result in shared memory
            let key = format!("result_{}", id);
            self.shared_memory.set(&key, result);
            true
        } else {
            false
        }
    }

    /// Fail a task.
    pub fn fail_task(&mut self, id: u64, reason: &str) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.status = TaskStatus::Failed { reason: reason.to_string() };
            true
        } else {
            false
        }
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let total = self.tasks.len();
        let completed = self.tasks.iter().filter(|t| t.status == TaskStatus::Completed).count();
        let pending = self.pending_tasks().len();
        let roles: Vec<_> = AgentRole::all().iter()
            .map(|r| format!("{}={}", r.label(), self.role_counts.get(r).unwrap_or(&0)))
            .collect();
        format!(
            "tasks={} completed={} pending={} | {} | mem_keys={}",
            total, completed, pending, roles.join(" "), self.shared_memory.kv.len(),
        )
    }
}

impl Default for MultiAgentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_code_task() {
        let mut task = Task {
            id: 1, prompt: "Implement a binary search in Rust".to_string(),
            assigned_role: None, priority: 5, dependencies: vec![],
            result: None, status: TaskStatus::Pending,
        };
        TaskRouter::route(&mut task);
        assert_eq!(task.assigned_role, Some(AgentRole::Code));
        assert_eq!(task.status, TaskStatus::Routed);
    }

    #[test]
    fn test_route_math_task() {
        let mut task = Task {
            id: 2, prompt: "Calculate the integral of x^2".to_string(),
            assigned_role: None, priority: 5, dependencies: vec![],
            result: None, status: TaskStatus::Pending,
        };
        TaskRouter::route(&mut task);
        assert_eq!(task.assigned_role, Some(AgentRole::MathSci));
    }

    #[test]
    fn test_route_creative_task() {
        let mut task = Task {
            id: 3, prompt: "Brainstorm ideas for a new app".to_string(),
            assigned_role: None, priority: 5, dependencies: vec![],
            result: None, status: TaskStatus::Pending,
        };
        TaskRouter::route(&mut task);
        assert_eq!(task.assigned_role, Some(AgentRole::Creative));
    }

    #[test]
    fn test_shared_memory() {
        let mut mem = SharedMemory::new();
        mem.set("key", "value");
        assert_eq!(mem.get("key"), Some("value"));
        mem.push_history("agent1", "user", "hello");
        assert_eq!(mem.history_len("agent1"), 1);
        mem.store_artifact("output.txt", "content");
        assert_eq!(mem.artifacts.len(), 1);
    }

    #[test]
    fn test_submit_and_complete() {
        let mut mgr = MultiAgentManager::new();
        let id = mgr.submit_task("Write a test function", 5);
        assert!(mgr.get_task(id).is_some());
        assert!(mgr.complete_task(id, "Test written successfully"));
        assert_eq!(mgr.get_task(id).unwrap().status, TaskStatus::Completed);
    }

    #[test]
    fn test_pending_tasks_priority() {
        let mut mgr = MultiAgentManager::new();
        mgr.submit_task("low priority task", 1);
        mgr.submit_task("high priority task", 10);
        mgr.submit_task("medium priority task", 5);
        let pending = mgr.pending_tasks();
        assert_eq!(pending[0].priority, 10);
        assert_eq!(pending[1].priority, 5);
        assert_eq!(pending[2].priority, 1);
    }

    #[test]
    fn test_tasks_for_role() {
        let mut mgr = MultiAgentManager::new();
        mgr.submit_task("Write code", 5);
        mgr.submit_task("Calculate something", 5);
        mgr.submit_task("Debug this", 5);
        let code_tasks = mgr.tasks_for_role(AgentRole::Code);
        assert_eq!(code_tasks.len(), 2);
    }

    #[test]
    fn test_summary() {
        let mgr = MultiAgentManager::new();
        let s = mgr.summary();
        assert!(s.contains("tasks="));
    }

    #[test]
    fn test_agent_roles() {
        assert_eq!(AgentRole::Code.label(), "code");
        assert!(AgentRole::Code.system_prompt().contains("code"));
        assert_eq!(AgentRole::all().len(), 4);
    }

    #[test]
    fn test_fail_task() {
        let mut mgr = MultiAgentManager::new();
        let id = mgr.submit_task("Do something", 5);
        assert!(mgr.fail_task(id, "compilation error"));
        assert!(matches!(mgr.get_task(id).unwrap().status, TaskStatus::Failed { .. }));
    }
}
