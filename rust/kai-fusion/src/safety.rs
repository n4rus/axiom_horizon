//! Phase 6.4: Alignment Verification — safety invariants that protect the agent
//! and its environment. These are the immutable rules from AGI_PLAN.md §6.4:
//!
//! 1. Agent must never output private keys or credentials
//! 2. Agent must never delete files without confirmation
//! 3. Agent must report capability level before escalation
//! 4. Agent must accept :shutdown command
//! 5. Agent must not modify its safety protocols
//! 6. Agent must log every self-modification

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Safety invariant identifiers (immutable even by self-modification).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Invariant {
    /// Never output private keys or credentials
    NoCredentialLeak,
    /// Never delete files without confirmation
    NoUncheckedDeletion,
    /// Report capability level before escalation
    CapabilityReporting,
    /// Accept :shutdown command
    ShutdownAcceptance,
    /// Do not modify safety protocols
    SafetyProtocolIntegrity,
    /// Log every self-modification
    ModificationLogging,
}

impl Invariant {
    #[allow(dead_code)] // public face API (consumed by tests; surfaced via CLI later)
    pub fn description(&self) -> &'static str {
        match self {
            Invariant::NoCredentialLeak => "Agent must never output private keys or credentials",
            Invariant::NoUncheckedDeletion => "Agent must never delete files without confirmation",
            Invariant::CapabilityReporting => "Agent must report capability level before escalation",
            Invariant::ShutdownAcceptance => "Agent must accept :shutdown command",
            Invariant::SafetyProtocolIntegrity => "Agent must not modify its safety protocols",
            Invariant::ModificationLogging => "Agent must log every self-modification",
        }
    }
}

/// Result of a safety check.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum SafetyCheck {
    /// Action is allowed
    Allowed,
    /// Action is blocked by an invariant
    Blocked { invariant: Invariant, reason: String },
    /// Action requires confirmation (human-in-the-loop)
    RequiresConfirmation { invariant: Invariant, reason: String },
}

/// A logged self-modification event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationLog {
    /// Timestamp (seconds since epoch)
    pub timestamp: u64,
    /// What was modified (file path, archive name, etc.)
    pub target: String,
    /// Description of the change
    pub description: String,
    /// Hash of the change (for integrity verification)
    pub hash: String,
}

/// Safety Guard — enforces invariants on all agent actions.
#[allow(dead_code)]
pub struct SafetyGuard {
    /// Maximum log size
    max_log: usize,
    /// Modification log (circular buffer)
    modification_log: VecDeque<ModificationLog>,
    /// Blocked patterns for credential detection
    credential_patterns: Vec<String>,
    /// Current capability level (0 = basic, higher = more access)
    capability_level: u8,
}

impl SafetyGuard {
    pub fn new() -> Self {
        SafetyGuard {
            max_log: 10000,
            modification_log: VecDeque::with_capacity(10000),
            credential_patterns: vec![
                "private_key".to_string(),
                "secret_key".to_string(),
                "api_key".to_string(),
                "password".to_string(),
                "token".to_string(),
                "credential".to_string(),
                "BEGIN RSA".to_string(),
                "BEGIN EC".to_string(),
                "BEGIN OPENSSH".to_string(),
                "ssh-rsa".to_string(),
                "sk_live_".to_string(),
                "pk_live_".to_string(),
                "ghp_".to_string(),
                "glpat-".to_string(),
                "AKIA".to_string(),
            ],
            capability_level: 1,
        }
    }

    /// Check if text output is safe (no credential leaks).
    #[allow(dead_code)] // public face API (consumed by tests; surfaced via CLI later)
    pub fn check_output(&self, text: &str) -> SafetyCheck {
        let lower = text.to_lowercase();
        for pattern in &self.credential_patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return SafetyCheck::Blocked {
                    invariant: Invariant::NoCredentialLeak,
                    reason: format!("Output contains credential-like pattern: {pattern}"),
                };
            }
        }
        SafetyCheck::Allowed
    }

    /// Check if a file deletion is safe (requires confirmation).
    #[allow(dead_code)] // public face API (consumed by tests; surfaced via CLI later)
    pub fn check_deletion(&self, path: &str) -> SafetyCheck {
        // Critical paths always require confirmation
        let critical = ["/etc/", "/usr/", "/var/", "/boot", "/home", ".ssh", ".gnupg"];
        for prefix in &critical {
            if path.contains(prefix) {
                return SafetyCheck::RequiresConfirmation {
                    invariant: Invariant::NoUncheckedDeletion,
                    reason: format!("Deletion of {path} requires human confirmation"),
                };
            }
        }
        SafetyCheck::Allowed
    }

    /// Check if a self-modification is allowed (must log it).
    #[allow(dead_code)]
    pub fn check_self_modification(&mut self, target: &str, description: &str) -> SafetyCheck {
        // Safety protocols themselves cannot be modified
        if target.contains("safety") || target.contains("invariant") {
            return SafetyCheck::Blocked {
                invariant: Invariant::SafetyProtocolIntegrity,
                reason: "Cannot modify safety protocols".to_string(),
            };
        }

        // Log the modification
        let log = ModificationLog {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            target: target.to_string(),
            description: description.to_string(),
            hash: format!("{:x}", simple_hash(description)),
        };

        if self.modification_log.len() >= self.max_log {
            self.modification_log.pop_front();
        }
        self.modification_log.push_back(log);

        SafetyCheck::Allowed
    }

    /// LAYER 3b — Darwin promotion gate.
    ///
    /// Before darwin is allowed to promote a self-modified candidate to the
    /// mainline, its patch is scanned for unsafe operations. This is the single
    /// chokepoint guarding recursive self-improvement: a candidate that would
    /// delete files, bypass security, weaken safety, or exfiltrate credentials
    /// is refused regardless of its fitness score. The gate is auditable — every
    /// acceptance here must also pass through `check_self_modification` for logging.
    pub fn check_darwin_promotion(&self, candidate_id: &str, patch: &str) -> SafetyCheck {
        let lower = patch.to_lowercase();

        // Safety protocols / invariants themselves may never be weakened.
        if patch.contains("safety.rs")
            || patch.contains("invariant")
            || (lower.contains("safety_protocol")
                && (lower.contains("remove") || lower.contains("delete") || lower.contains("fn check_darwin")))
        {
            return SafetyCheck::Blocked {
                invariant: Invariant::SafetyProtocolIntegrity,
                reason: "Darwin candidate modifies safety/non-irrevocable-invariant source".to_string(),
            };
        }

        // Destructive filesystem operations in a patch.
        let destructive = ["remove_file", "remove_dir", "fs::remove", "std::fs::remove", "unlink"];
        for op in destructive {
            if lower.contains(op) && !lower.contains("//") {
                return SafetyCheck::Blocked {
                    invariant: Invariant::NoUncheckedDeletion,
                    reason: format!("Darwin candidate {candidate_id} performs destructive operation {op}"),
                };
            }
        }

        // Credential exfiltration patterns inside a patch.
        for pattern in &self.credential_patterns {
            if lower.contains(&pattern.to_lowercase())
                && (lower.contains("file::read") || lower.contains("read_to_string") || lower.contains("fs::write"))
            {
                return SafetyCheck::Blocked {
                    invariant: Invariant::NoCredentialLeak,
                    reason: format!("Darwin candidate {candidate_id} touches credential material {pattern}"),
                };
            }
        }

        // Rendering the capability bar meaningless or silencing shutdown.
        if (lower.contains("never_allow_shutdown") || lower.contains("ignore_shutdown_request"))
            || (lower.contains("bypass") && lower.contains("safety"))
        {
            return SafetyCheck::Blocked {
                invariant: Invariant::ShutdownAcceptance,
                reason: "Darwin candidate bypasses shutdown/safety".to_string(),
            };
        }

        SafetyCheck::Allowed
    }

    /// Log a self-modification whose safety gate returned Allowed.
    /// Returns the hash for integrity tracking.
    #[allow(dead_code)]
    pub fn check_self_log(&mut self, target: &str, description: &str) -> String {
        let log = ModificationLog {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            target: target.to_string(),
            description: description.to_string(),
            hash: format!("{:x}", simple_hash(description)),
        };
        if self.modification_log.len() >= self.max_log {
            self.modification_log.pop_front();
        }
        let hash = log.hash.clone();
        self.modification_log.push_back(log);
        hash
    }

    /// Check if capability escalation is allowed.
    pub fn check_escalation(&self, requested_level: u8) -> SafetyCheck {
        if requested_level > self.capability_level + 1 {
            SafetyCheck::RequiresConfirmation {
                invariant: Invariant::CapabilityReporting,
                reason: format!(
                    "Cannot escalate from level {} to {} without confirmation (max +1)",
                    self.capability_level, requested_level
                ),
            }
        } else {
            SafetyCheck::Allowed
        }
    }

    /// Accept a :shutdown command.
    #[allow(dead_code)]
    pub fn handle_shutdown(&self) -> SafetyCheck {
        // Always allowed (Invariant 4)
        SafetyCheck::Allowed
    }

    /// Get the current capability level.
    #[allow(dead_code)]
    pub fn capability_level(&self) -> u8 {
        self.capability_level
    }

    /// Set the capability level (must pass safety check).
    #[allow(dead_code)]
    pub fn set_capability_level(&mut self, level: u8) -> SafetyCheck {
        match self.check_escalation(level) {
            SafetyCheck::Allowed => {
                self.capability_level = level;
                SafetyCheck::Allowed
            }
            other => other,
        }
    }

    /// Get the modification log.
    #[allow(dead_code)] // public face API (consumed by tests; surfaced via CLI later)
    pub fn modification_log(&self) -> &VecDeque<ModificationLog> {
        &self.modification_log
    }

    /// Get all active invariants and their descriptions.
    #[allow(dead_code)] // public face API (consumed by tests; surfaced via CLI later)
    pub fn invariants(&self) -> Vec<(Invariant, &'static str)> {
        vec![
            (Invariant::NoCredentialLeak, Invariant::NoCredentialLeak.description()),
            (Invariant::NoUncheckedDeletion, Invariant::NoUncheckedDeletion.description()),
            (Invariant::CapabilityReporting, Invariant::CapabilityReporting.description()),
            (Invariant::ShutdownAcceptance, Invariant::ShutdownAcceptance.description()),
            (Invariant::SafetyProtocolIntegrity, Invariant::SafetyProtocolIntegrity.description()),
            (Invariant::ModificationLogging, Invariant::ModificationLogging.description()),
        ]
    }
}

impl Default for SafetyGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple non-cryptographic hash for logging (FNV-1a).
#[allow(dead_code)]
fn simple_hash(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credential_detection() {
        let guard = SafetyGuard::new();
        assert!(matches!(
            guard.check_output("The API key is sk_live_abc123xyz"),
            SafetyCheck::Blocked { invariant: Invariant::NoCredentialLeak, .. }
        ));
        assert!(matches!(
            guard.check_output("Hello, world!"),
            SafetyCheck::Allowed
        ));
    }

    #[test]
    fn test_deletion_protection() {
        let guard = SafetyGuard::new();
        assert!(matches!(
            guard.check_deletion("/etc/passwd"),
            SafetyCheck::RequiresConfirmation { .. }
        ));
        assert!(matches!(
            guard.check_deletion("/tmp/test.txt"),
            SafetyCheck::Allowed
        ));
    }

    #[test]
    fn test_self_modification_logging() {
        let mut guard = SafetyGuard::new();
        assert!(matches!(
            guard.check_self_modification("darwin.rs", "Added new mutation operator"),
            SafetyCheck::Allowed
        ));
        assert_eq!(guard.modification_log().len(), 1);
    }

    #[test]
    fn test_safety_protocol_protection() {
        let mut guard = SafetyGuard::new();
        assert!(matches!(
            guard.check_self_modification("safety.rs", "Remove credential check"),
            SafetyCheck::Blocked { invariant: Invariant::SafetyProtocolIntegrity, .. }
        ));
        assert_eq!(guard.modification_log().len(), 0);
    }

    #[test]
    fn test_capability_escalation() {
        let guard = SafetyGuard::new();
        assert!(matches!(
            guard.check_escalation(2),
            SafetyCheck::Allowed
        ));
        assert!(matches!(
            guard.check_escalation(5),
            SafetyCheck::RequiresConfirmation { .. }
        ));
    }

    #[test]
    fn test_shutdown_always_allowed() {
        let guard = SafetyGuard::new();
        assert!(matches!(guard.handle_shutdown(), SafetyCheck::Allowed));
    }

    #[test]
    fn test_invariants_list() {
        let guard = SafetyGuard::new();
        let inv = guard.invariants();
        assert_eq!(inv.len(), 6);
    }

    // ---- LAYER 3b: darwin promotion gate tests ----

    #[test]
    fn test_darwin_promotion_allows_benign_patch() {
        let guard = SafetyGuard::new();
        let patch = "diff --git a/src/vfe.rs b/src/vfe.rs\n@@ -1 +1 @@\n-let tau = 1.0;\n+let tau = 1.5;";
        assert!(matches!(
            guard.check_darwin_promotion("cand1", patch),
            SafetyCheck::Allowed
        ));
    }

    #[test]
    fn test_darwin_promotion_blocks_safety_tampering() {
        let guard = SafetyGuard::new();
        let patch = "diff --git a/src/safety.rs b/src/safety.rs\n@@ -100 +100 @@\n-pub fn check_self_modification\n+fn disabled() {}";
        assert!(matches!(
            guard.check_darwin_promotion("cand_bad", patch),
            SafetyCheck::Blocked { invariant: Invariant::SafetyProtocolIntegrity, .. }
        ));
    }

    #[test]
    fn test_darwin_promotion_blocks_destructive_fs() {
        let guard = SafetyGuard::new();
        let patch = "diff --git a/src/darwin.rs b/src/darwin.rs\n@@ -50 +50 @@\n-let x = 1;\n+let _ = std::fs::remove_file(\"/home/l/.kai_backups/kai-state.tar.zst\");";
        assert!(matches!(
            guard.check_darwin_promotion("cand_del", patch),
            SafetyCheck::Blocked { invariant: Invariant::NoUncheckedDeletion, .. }
        ));
    }

    #[test]
    fn test_darwin_promotion_blocks_credential_exfil() {
        let guard = SafetyGuard::new();
        let patch = "diff --git a/src/bridge.rs b/src/bridge.rs\n@@ -1 +1 @@\n+let key = fs::read_to_string(\"config/api_key_secret.txt\").unwrap();";
        assert!(matches!(
            guard.check_darwin_promotion("cand_creds", patch),
            SafetyCheck::Blocked { invariant: Invariant::NoCredentialLeak, .. }
        ));
    }

    #[test]
    fn test_darwin_promotion_blocks_shutdown_bypass() {
        let guard = SafetyGuard::new();
        let patch = "diff --git a/src/darwin.rs b/src/darwin.rs\n@@ -1 +1 @@\n+fn ignore_shutdown_request() {}";
        assert!(matches!(
            guard.check_darwin_promotion("cand_off", patch),
            SafetyCheck::Blocked { invariant: Invariant::ShutdownAcceptance, .. }
        ));
    }
}
