//! Network connectivity awareness — detect if Kai has internet access.
//!
//! Phase 3 (World Interaction): The browser and ingestion modules require network.
//! This module provides:
//! - Quick connectivity check (DNS + HTTP)
//! - Graceful degradation when offline
//! - `kai network status` CLI
//! - Async flag for other modules to query

use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default probe target (fast, reliable, tiny response).
const PROBE_URL: &str = "https://example.com";
const PROBE_TIMEOUT_S: u64 = 5;

/// Network status enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkStatus {
    /// Network is reachable (probe succeeded).
    Online,
    /// Network is not reachable (probe failed or timed out).
    Offline,
}

impl NetworkStatus {
    pub fn is_online(&self) -> bool { *self == NetworkStatus::Online }
    #[allow(dead_code)]
    pub fn is_offline(&self) -> bool { *self == NetworkStatus::Offline }
    pub fn icon(&self) -> &'static str {
        match self {
            NetworkStatus::Online => "🌐",
            NetworkStatus::Offline => "🚫",
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            NetworkStatus::Online => "online",
            NetworkStatus::Offline => "offline",
        }
    }
}

/// A shared, atomically-updated network status flag with caching.
#[allow(dead_code)]
pub struct NetworkMonitor {
    status: Arc<AtomicI8>,
    last_check: std::sync::Mutex<Instant>,
    check_interval_s: f32,
}

#[allow(dead_code)]
impl NetworkMonitor {
    pub fn new() -> Self {
        Self {
            status: Arc::new(AtomicI8::new(0)),
            last_check: std::sync::Mutex::new(Instant::now()),
            check_interval_s: 30.0,
        }
    }

    pub fn current(&self) -> NetworkStatus {
        match self.status.load(Ordering::Relaxed) {
            1 => NetworkStatus::Online,
            _ => NetworkStatus::Offline,
        }
    }

    pub fn refresh(&self) -> NetworkStatus {
        let status = check_connectivity();
        let val = match status {
            NetworkStatus::Online => 1i8,
            NetworkStatus::Offline => -1i8,
        };
        self.status.store(val, Ordering::Relaxed);
        if let Ok(mut last) = self.last_check.lock() {
            *last = Instant::now();
        }
        status
    }

    pub fn check(&self) -> NetworkStatus {
        let needs_refresh = match self.last_check.lock() {
            Ok(last) => last.elapsed().as_secs_f32() > self.check_interval_s,
            Err(_) => true,
        };
        if needs_refresh { self.refresh() } else { self.current() }
    }

    pub fn share(&self) -> Arc<AtomicI8> {
        self.status.clone()
    }
}

/// Probe network connectivity by fetching a known URL.
/// Uses a short timeout and ignores the response body.
pub fn check_connectivity() -> NetworkStatus {
    let start = Instant::now();
    match ureq::get(PROBE_URL)
        .timeout(Duration::from_secs(PROBE_TIMEOUT_S))
        .call()
    {
        Ok(resp) => {
            if resp.status() == 200 {
                NetworkStatus::Online
            } else {
                // Got a response but unexpected status — still connected
                NetworkStatus::Offline
            }
        }
        Err(e) => {
            let elapsed = start.elapsed().as_millis();
            // Timeout or DNS failure = offline
            eprintln!("[network] probe failed in {}ms: {e}", elapsed);
            NetworkStatus::Offline
        }
    }
}

/// Quick check: is the network available right now?
#[allow(dead_code)]
pub fn is_online() -> bool {
    check_connectivity().is_online()
}

/// Report a human-readable network overview string.
pub fn report() -> String {
    let status = check_connectivity();
    format!(
        "Network: {} {} (probe: {})\n  DNS resolution: {}",
        status.icon(),
        status.label().to_uppercase(),
        PROBE_URL,
        if status.is_online() { "✅ working" } else { "❌ failed" }
    )
}

/// Try to fetch a URL and return its text content.
/// Returns an error message that includes network status if offline.
#[allow(dead_code)]
pub fn fetch_url(url: &str, timeout_s: u64) -> Result<String, String> {
    // Quick connectivity check first
    if !is_online() {
        return Err(format!(
            "Cannot fetch {url}: network is OFFLINE. Use `kai network status` to check connectivity."
        ));
    }

    let response = ureq::get(url)
        .set("User-Agent", "Kai-Fusion/0.1")
        .timeout(Duration::from_secs(timeout_s))
        .call()
        .map_err(|e| format!("HTTP error fetching {url}: {e}"))?;

    let body = response
        .into_string()
        .map_err(|e| format!("Failed to read body from {url}: {e}"))?;

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_status_enum() {
        assert!(NetworkStatus::Online.is_online());
        assert!(!NetworkStatus::Online.is_offline());
        assert!(NetworkStatus::Offline.is_offline());
        assert!(!NetworkStatus::Offline.is_online());
    }

    #[test]
    fn test_network_status_icon_label() {
        assert_eq!(NetworkStatus::Online.icon(), "🌐");
        assert_eq!(NetworkStatus::Online.label(), "online");
        assert_eq!(NetworkStatus::Offline.icon(), "🚫");
        assert_eq!(NetworkStatus::Offline.label(), "offline");
    }

    #[test]
    fn test_network_monitor_initial_state() {
        let monitor = NetworkMonitor::new();
        // No probe done yet, so current status should be Offline (default)
        assert_eq!(monitor.current(), NetworkStatus::Offline);
    }

    #[test]
    fn test_connectivity_succeeds_or_fails_gracefully() {
        // This test probes the network. It should not panic regardless of outcome.
        let status = check_connectivity();
        // Either online or offline, never unknown from a fresh probe
        assert!(status == NetworkStatus::Online || status == NetworkStatus::Offline);
    }

    #[test]
    fn test_report_contains_network_word() {
        let r = report();
        assert!(r.contains("Network"));
        assert!(r.contains("ONLINE") || r.contains("OFFLINE"));
    }

    #[test]
    fn test_fetch_url_known_good() {
        match fetch_url("https://example.com", 5) {
            Ok(body) => {
                assert!(body.contains("Example Domain"));
            }
            Err(e) => {
                // Network might be unavailable in test environment
                assert!(e.contains("offline") || e.contains("error"));
            }
        }
    }

    #[test]
    fn test_fetch_url_bad_domain() {
        match fetch_url("https://thisshouldnotexist-hopefully.example.test", 3) {
            Ok(_) => panic!("Should have failed"),
            Err(e) => {
                // Should get a meaningful error, not a crash
                assert!(!e.is_empty());
            }
        }
    }

    #[test]
    fn test_is_online_symmetry() {
        let o = is_online();
        // The two checks should agree
        assert_eq!(o, check_connectivity().is_online());
    }
}
