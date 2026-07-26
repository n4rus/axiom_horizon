//! ESP32 PoGIE bridge — physical integration for Phase 7 (Singularity).
//!
//! PoGIE = Process of Global Intelligence Evolution
//!
//! Every home runs an ESP32-S3 with PoGIE firmware:
//! - Connects to local AI node (ollama on consumer hardware)
//! - Reports energy availability via serial/UART
//! - Controls actuators (lights, HVAC, appliances) based on AI decisions
//! - Participates in decentralized attractor network
//!
//! This module provides the protocol definitions and stubs for:
//! - Energy grid monitoring and allocation
//! - Serial bridge to ESP32-S3 devices
//! - Node discovery and status reporting
//!
//! When hardware is connected, the serial bridge sends/receives
//! structured commands over UART at 115200 baud.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── PoGIE Protocol Constants ──────────────────────────────────────────────

/// Default UART baud rate for ESP32-S3 PoGIE bridge.
pub const DEFAULT_BAUD_RATE: u32 = 115_200;

/// Default serial port for ESP32-S3 on Linux.
pub const DEFAULT_PORT: &str = "/dev/ttyUSB0";

/// Maximum message length over serial (bytes).
pub const MAX_MESSAGE_LEN: usize = 4096;

/// Heartbeat interval for ESP32 nodes (seconds).
pub const HEARTBEAT_INTERVAL_S: f32 = 10.0;

/// Energy report interval (seconds).
pub const ENERGY_REPORT_INTERVAL_S: f32 = 60.0;

// ── PoGIE Node Identity ───────────────────────────────────────────────────

/// Role of a node in the PoGIE network.
#[derive(Debug, Clone, PartialEq)]
pub enum PogieNodeRole {
    /// Coordinator: manages resource allocation for the local network.
    Coordinator,
    /// Inference: runs local AI models (ollama).
    Inference,
    /// Storage: hosts model weights and vector attractors.
    Storage,
    /// Actuator: controls physical devices (lights, HVAC, appliances).
    Actuator,
    /// Sensor: reports environmental data (temperature, humidity, power).
    Sensor,
}

impl PogieNodeRole {
    pub fn icon(&self) -> &'static str {
        match self {
            PogieNodeRole::Coordinator => "🗺️",
            PogieNodeRole::Inference => "🧠",
            PogieNodeRole::Storage => "💾",
            PogieNodeRole::Actuator => "⚙️",
            PogieNodeRole::Sensor => "📡",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            PogieNodeRole::Coordinator => "Coordinator",
            PogieNodeRole::Inference => "Inference",
            PogieNodeRole::Storage => "Storage",
            PogieNodeRole::Actuator => "Actuator",
            PogieNodeRole::Sensor => "Sensor",
        }
    }
}

/// Current status of a PoGIE node.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Online,
    Idle,
    Busy,
    Offline,
}

impl NodeStatus {
    pub const fn as_str(&self) -> &'static str {
        match self {
            NodeStatus::Online => "online",
            NodeStatus::Idle => "idle",
            NodeStatus::Busy => "busy",
            NodeStatus::Offline => "offline",
        }
    }
}

/// A PoGIE network node (typically an ESP32-S3 device).
#[derive(Debug, Clone)]
pub struct PogieNode {
    pub node_id: String,
    pub role: PogieNodeRole,
    pub compute_capacity: f32, // normalized FLOPS [0.0..1.0]
    pub memory_gb: f32,
    pub status: NodeStatus,
    pub last_heartbeat: Option<Instant>,
    pub ip_address: Option<String>,
}

impl PogieNode {
    pub fn new(node_id: &str, role: PogieNodeRole) -> Self {
        Self {
            node_id: node_id.to_string(),
            role,
            compute_capacity: 0.0,
            memory_gb: 0.0,
            status: NodeStatus::Offline,
            last_heartbeat: None,
            ip_address: None,
        }
    }

    pub fn is_available(&self) -> bool {
        self.status == NodeStatus::Online || self.status == NodeStatus::Idle
    }

    pub fn utilization(&self) -> f32 {
        if self.status == NodeStatus::Busy {
            1.0
        } else if self.status == NodeStatus::Idle {
            0.3
        } else {
            0.0
        }
    }
}

// ── Energy Grid ───────────────────────────────────────────────────────────

/// Energy grid monitor — tracks power availability and allocation.
#[derive(Debug, Clone)]
pub struct EnergyGrid {
    pub total_capacity_w: f32,
    pub current_draw_w: f32,
    pub renewable_w: f32, // solar/wind contribution
    pub grid_available_w: f32,
    pub allocation_map: HashMap<String, f32>, // node_id -> watts allocated
    pub price_per_kwh: f32, // local energy price
}

impl EnergyGrid {
    pub fn new(total_capacity_w: f32) -> Self {
        Self {
            total_capacity_w,
            current_draw_w: 0.0,
            renewable_w: 0.0,
            grid_available_w: total_capacity_w,
            allocation_map: HashMap::new(),
            price_per_kwh: 0.12, // default $0.12/kWh
        }
    }

    /// Available watts: capacity minus current draw.
    pub fn available_w(&self) -> f32 {
        self.total_capacity_w - self.current_draw_w
    }

    /// Attempt to allocate watts to a node. Returns true if successful.
    pub fn allocate(&mut self, node_id: &str, watts: f32) -> bool {
        if watts <= self.available_w() {
            self.current_draw_w += watts;
            self.allocation_map.insert(node_id.to_string(), watts);
            true
        } else {
            false
        }
    }

    /// Release watts previously allocated to a node.
    pub fn release(&mut self, node_id: &str) -> f32 {
        if let Some(w) = self.allocation_map.remove(node_id) {
            self.current_draw_w -= w;
            w
        } else {
            0.0
        }
    }

    /// Renewable energy percentage.
    pub fn renewable_ratio(&self) -> f32 {
        if self.total_capacity_w > 0.0 {
            self.renewable_w / self.total_capacity_w
        } else {
            0.0
        }
    }

    /// Generate a status report string.
    pub fn status_report(&self) -> String {
        format!(
            "⚡ Energy Grid: {:.1}/{:.1} W used | {:.1} W available | \
             {:.0}% renewable | price: ${:.2}/kWh",
            self.current_draw_w,
            self.total_capacity_w,
            self.available_w(),
            self.renewable_ratio() * 100.0,
            self.price_per_kwh,
        )
    }
}

// ── Serial Bridge (ESP32 ↔ Host) ─────────────────────────────────────────

/// Serial bridge to ESP32-S3 PoGIE device.
///
/// In the full build, this would use `serialport` crate for real UART.
/// Currently provides the protocol structure for when hardware is connected.
#[derive(Debug, Clone)]
pub struct Esp32Bridge {
    pub port: String,
    pub baud_rate: u32,
    pub connected: bool,
    pub last_response: Option<String>,
    pub pending_commands: Vec<String>,
}

impl Esp32Bridge {
    pub fn new(port: &str, baud_rate: u32) -> Self {
        Self {
            port: port.to_string(),
            baud_rate,
            connected: false,
            last_response: None,
            pending_commands: Vec::new(),
        }
    }

    /// Default bridge (uses DEFAULT_PORT and DEFAULT_BAUD_RATE).
    pub fn default() -> Self {
        Self::new(DEFAULT_PORT, DEFAULT_BAUD_RATE)
    }

    /// Connect to the ESP32 device (opens serial port).
    ///
    /// Returns Ok(true) if connected, Ok(false) if no hardware present.
    pub fn connect(&mut self) -> Result<bool, String> {
        #[cfg(feature = "serial")]
        {
            use serialport::SerialPort;
            let port = serialport::new(&self.port, self.baud_rate)
                .timeout(Duration::from_millis(100))
                .open();
            match port {
                Ok(_) => {
                    self.connected = true;
                    Ok(true)
                }
                Err(serialport::Error:: NoSuchDevice) => Ok(false),
                Err(e) => Err(format!("serial open error: {e}")),
            }
        }
        #[cfg(not(feature = "serial"))]
        {
            // Without serial feature, always report not connected.
            self.connected = false;
            Ok(false)
        }
    }

    /// Send a command to ESP32 and wait for response.
    pub fn send_command(&mut self, cmd: &str) -> Result<String, String> {
        if !self.connected {
            return Err("ESP32 not connected (no serial feature or no hardware)".to_string());
        }
        if cmd.len() > MAX_MESSAGE_LEN {
            return Err(format!("command too long ({} > {})", cmd.len(), MAX_MESSAGE_LEN));
        }
        // Protocol: send command as JSON, receive JSON response
        let _msg = format!("{{\"cmd\":\"{}\"}}\n", cmd);
        self.last_response = Some(format!("ack:{}", cmd));
        Ok(self.last_response.clone().unwrap())
    }

    /// Read a sensor value from ESP32.
    pub fn read_sensor(&mut self, sensor: &str) -> Result<f32, String> {
        self.send_command(&format!("read:{}", sensor))?;
        // Parse f32 from response (mock: return 0.0 when not connected)
        if !self.connected {
            return match sensor {
                "temperature" => Ok(22.5),  // mock ambient
                "humidity" => Ok(45.0),     // mock humidity
                "power_w" => Ok(12.0),      // mock power draw
                "voltage" => Ok(3.3),       // mock voltage
                _ => Ok(0.0),
            };
        }
        // When connected, parse real sensor data from response
        self.last_response
            .as_ref()
            .and_then(|r| r.split(':').nth(1))
            .and_then(|v| v.trim().parse::<f32>().ok())
            .ok_or_else(|| format!("cannot read sensor '{}'", sensor))
    }

    /// Send energy allocation command to ESP32.
    pub fn set_power_limit(&mut self, watts: f32) -> Result<String, String> {
        self.send_command(&format!("power_limit:{:.1}", watts))
    }

    /// Toggle an actuator on the ESP32.
    pub fn toggle_actuator(&mut self, pin: u8, state: bool) -> Result<String, String> {
        self.send_command(&format!("gpio:{}:{}", pin, if state { "HIGH" } else { "LOW" }))
    }
}

// ── PoGIE Protocol (Decentralized Resource Allocation) ────────────────────

/// The PoGIE protocol manages decentralized resource allocation across nodes.
#[derive(Debug, Clone)]
pub struct PogieProtocol {
    pub nodes: HashMap<String, PogieNode>,
    pub energy_grid: EnergyGrid,
    pub attractor_path: String,
}

impl PogieProtocol {
    pub fn new(attractor_path: &str, grid_capacity_w: f32) -> Self {
        Self {
            nodes: HashMap::new(),
            energy_grid: EnergyGrid::new(grid_capacity_w),
            attractor_path: attractor_path.to_string(),
        }
    }

    /// Register a node in the PoGIE network.
    pub fn register_node(&mut self, node: PogieNode) {
        self.nodes.insert(node.node_id.clone(), node);
    }

    /// Remove a node from the network.
    pub fn deregister_node(&mut self, node_id: &str) -> Option<PogieNode> {
        self.nodes.remove(node_id)
    }

    /// Get the best available inference node (highest compute, lowest utilization).
    pub fn best_inference_node(&self) -> Option<&PogieNode> {
        self.nodes
            .values()
            .filter(|n| n.role == PogieNodeRole::Inference && n.is_available())
            .max_by(|a, b| {
                a.compute_capacity
                    .partial_cmp(&b.compute_capacity)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(
                        a.utilization()
                            .partial_cmp(&b.utilization())
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
            })
    }

    /// Allocate compute + energy for an inference task.
    /// Returns (node_id, watts_allocated) or None if insufficient resources.
    pub fn allocate_task(&mut self, task_flops: f32) -> Option<(String, f32)> {
        let node_id = self.best_inference_node()?.node_id.clone();
        let watts = task_flops * 5.0; // rough FLOPS-to-watts estimate
        if self.energy_grid.allocate(&node_id, watts) {
            Some((node_id, watts))
        } else {
            None
        }
    }

    /// Generate a network-wide status report.
    pub fn network_report(&self) -> String {
        let mut report = String::new();
        report.push_str("╔══════════════════════════════════════════╗\n");
        report.push_str("║         PoGIE Network Status              ║\n");
        report.push_str("╠══════════════════════════════════════════╣\n");
        report.push_str(&format!("  Nodes online: {}/{}\n",
            self.nodes.values().filter(|n| n.status == NodeStatus::Online).count(),
            self.nodes.len()));
        for node in self.nodes.values() {
            report.push_str(&format!(
                "  {} {} [{}] {:.1} GFLOPS | {:.1} GB | {}\n",
                node.role.icon(),
                node.node_id,
                node.status.as_str(),
                node.compute_capacity * 100.0,
                node.memory_gb,
                if node.is_available() { "✓ available" } else { "✗ busy" }
            ));
        }
        report.push_str(&format!(
            "\n  {}\n",
            self.energy_grid.status_report()
        ));

        // Convergence info
        report.push_str(&format!(
            "  Attractor: {} (PoGIE endpoint)\n",
            self.attractor_path
        ));
        report.push_str("╚══════════════════════════════════════════╝\n");
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pogie_node_creation() {
        let node = PogieNode::new("esp32-01", PogieNodeRole::Inference);
        assert_eq!(node.node_id, "esp32-01");
        assert_eq!(node.role, PogieNodeRole::Inference);
        assert_eq!(node.status, NodeStatus::Offline);
        assert!(!node.is_available());
    }

    #[test]
    fn test_node_status_icon() {
        assert_eq!(PogieNodeRole::Inference.icon(), "🧠");
        assert_eq!(PogieNodeRole::Actuator.icon(), "⚙️");
        assert_eq!(PogieNodeRole::Coordinator.icon(), "🗺️");
    }

    #[test]
    fn test_node_utilization() {
        let mut node = PogieNode::new("esp32-02", PogieNodeRole::Actuator);
        assert_eq!(node.utilization(), 0.0);
        node.status = NodeStatus::Idle;
        assert_eq!(node.utilization(), 0.3);
        node.status = NodeStatus::Busy;
        assert_eq!(node.utilization(), 1.0);
    }

    #[test]
    fn test_energy_grid_allocation() {
        let mut grid = EnergyGrid::new(100.0);
        assert_eq!(grid.available_w(), 100.0);
        assert!(grid.allocate("inference-1", 30.0));
        assert_eq!(grid.available_w(), 70.0);
        assert!(grid.allocate("inference-2", 80.0) == false); // not enough
        let freed = grid.release("inference-1");
        assert_eq!(freed, 30.0);
        assert_eq!(grid.available_w(), 100.0);
    }

    #[test]
    fn test_esp32_bridge_default() {
        let bridge = Esp32Bridge::default();
        assert_eq!(bridge.port, DEFAULT_PORT);
        assert_eq!(bridge.baud_rate, DEFAULT_BAUD_RATE);
        assert!(!bridge.connected);
    }

    #[test]
    fn test_energy_grid_renewable_ratio() {
        let mut grid = EnergyGrid::new(100.0);
        grid.renewable_w = 40.0;
        assert_eq!(grid.renewable_ratio(), 0.4);
    }

    #[test]
    fn test_pogie_protocol_register_node() {
        let mut proto = PogieProtocol::new(".axiom_state/kai_fusion_attractor.json", 200.0);
        let node = PogieNode::new("esp32-01", PogieNodeRole::Inference);
        proto.register_node(node);
        assert_eq!(proto.nodes.len(), 1);
        assert!(proto.best_inference_node().is_none()); // offline
    }
}
