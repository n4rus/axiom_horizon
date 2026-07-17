//! Kai's Causal World Model (Structural Causal Model in Engram)
//! 
//! Implements a Structural Causal Model (SCM) over the Engram latent space.
//! Nodes represent entities/states/actions; edges represent causal mechanisms.
//! Provides the do-operator for intervention: P(Y | do(X=x)).
//! 
//! Based on the Tonal Collapse framework: the attention manifold g_ij = 1 - a_ij
//! defines the causal geometry. Interventions are geodesic perturbations.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

/// Node types in the causal graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Entity,      // Objects, people, concepts
    State,       // Properties, conditions
    Action,      // Interventions, operations
    Observation, // Sensor readings
}

/// A node in the causal graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    pub id: String,
    pub node_type: NodeType,
    /// Latent embedding in Engram space (768-dim)
    pub embedding: Vec<f32>,
    /// Current value/state
    pub value: NodeValue,
    /// Confidence in this node's state (0.0 - 1.0)
    pub confidence: f32,
    /// Timestamp of last update
    pub timestamp: u64,
}

/// Value held by a causal node
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeValue {
    Boolean(bool),
    Numeric(f32),
    Categorical(String),
    Vector(Vec<f32>),
    Text(String),
}

/// A causal mechanism (structural equation): X = f(parents, noise)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalMechanism {
    pub child: String,
    pub parents: Vec<String>,
    /// Mechanism type
    pub mech_type: MechanismType,
    /// Parameters (weights, thresholds, etc.)
    pub params: HashMap<String, f32>,
    /// Noise variance (epistemic uncertainty)
    pub noise_var: f32,
}

/// Types of causal mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MechanismType {
    Linear,       // X = sum(w_i * parent_i) + noise
    Threshold,    // X = 1 if sum(w_i * parent_i) > threshold else 0
    Categorical,  // X = categorical distribution over parents
    Functional,   // X = f(parents) where f is a learned neural net (simplified)
    Copy,         // X = parent (identity)
}

impl Default for MechanismType {
    fn default() -> Self {
        MechanismType::Linear
    }
}

/// The Structural Causal Model (SCM) - Kai's world model
#[allow(dead_code)]
pub struct StructuralCausalModel {
    nodes: Arc<RwLock<HashMap<String, CausalNode>>>,
    mechanisms: Arc<RwLock<HashMap<String, CausalMechanism>>>,
    /// Adjacency: child -> parents
    graph: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    /// Reverse adjacency: parent -> children
    rev_graph: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    /// Engram memory for latent embeddings
    engram: Arc<dyn EngramInterface>,
    /// Intervention history for learning
    intervention_log: Arc<RwLock<Vec<InterventionRecord>>>,
}

/// Interface to Engram memory (to be implemented by the host)
pub trait EngramInterface: Send + Sync {
    /// Get embedding for a concept
    fn get_embedding(&self, concept: &str) -> Option<Vec<f32>>;
    /// Store embedding for a concept
    #[allow(dead_code)]
    fn store_embedding(&self, concept: &str, embedding: Vec<f32>);
    /// Query similar concepts by embedding
    #[allow(dead_code)]
    fn query_similar(&self, embedding: &[f32], k: usize) -> Vec<(String, f32)>;
}

/// Record of an intervention for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionRecord {
    pub timestamp: u64,
    pub target: String,
    pub value: NodeValue,
    pub predicted_outcome: HashMap<String, NodeValue>,
    pub actual_outcome: Option<HashMap<String, NodeValue>>,
    pub vfe_before: f32,
    pub vfe_after: Option<f32>,
}

#[allow(dead_code)]
impl StructuralCausalModel {
    pub fn new(engram: Arc<dyn EngramInterface>) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            mechanisms: Arc::new(RwLock::new(HashMap::new())),
            graph: Arc::new(RwLock::new(HashMap::new())),
            rev_graph: Arc::new(RwLock::new(HashMap::new())),
            engram,
            intervention_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add or update a node in the causal graph
    pub fn add_node(&self, node: CausalNode) {
        let mut nodes = self.nodes.write().unwrap();
        let mut graph = self.graph.write().unwrap();
        let mut rev_graph = self.rev_graph.write().unwrap();

        let id = node.id.clone();
        nodes.insert(id.clone(), node);
        graph.entry(id.clone()).or_default();
        rev_graph.entry(id).or_default();
    }

    /// Add a causal mechanism (structural equation)
    pub fn add_mechanism(&self, mechanism: CausalMechanism) {
        let mut mechanisms = self.mechanisms.write().unwrap();
        let mut graph = self.graph.write().unwrap();
        let mut rev_graph = self.rev_graph.write().unwrap();

        let child = mechanism.child.clone();
        let parents = mechanism.parents.clone();

        // Register mechanism
        mechanisms.insert(child.clone(), mechanism);

        // Update graph structure
        graph.insert(child.clone(), parents.iter().cloned().collect());
        for parent in parents {
            rev_graph.entry(parent).or_default().insert(child.clone());
        }
    }

    /// Get node by ID
    pub fn get_node(&self, id: &str) -> Option<CausalNode> {
        self.nodes.read().unwrap().get(id).cloned()
    }

    /// Get all descendants of a node (transitive closure)
    pub fn get_descendants(&self, node: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![node.to_string()];
        
        while let Some(current) = stack.pop() {
            if visited.insert(current.clone()) {
                if let Some(children) = self.rev_graph.read().unwrap().get(&current) {
                    for child in children {
                        stack.push(child.clone());
                    }
                }
            }
        }
        visited.remove(node);
        visited
    }

    /// Get all ancestors of a node (transitive closure)
    pub fn get_ancestors(&self, node: &str) -> HashSet<String> {
        let graph = self.graph.read().unwrap();
        let mut visited = HashSet::new();
        let mut stack = vec![node.to_string()];
        
        while let Some(current) = stack.pop() {
            if visited.insert(current.clone()) {
                if let Some(parents) = graph.get(&current) {
                    for parent in parents {
                        stack.push(parent.clone());
                    }
                }
            }
        }
        visited.remove(node);
        visited
    }

    /// Check if there's a causal path from A to B
    pub fn has_causal_path(&self, from: &str, to: &str) -> bool {
        self.get_descendants(from).contains(to)
    }

    /// The DO-OPERATOR: Intervene on a node, compute post-intervention distribution
    /// 
    /// do(X = x) means: remove all incoming edges to X, set X = x, propagate forward
    pub fn do_intervention(
        &self,
        target: &str,
        value: NodeValue,
        context: Option<HashMap<String, NodeValue>>,
    ) -> Result<HashMap<String, NodeValue>, String> {
        let nodes = self.nodes.write().unwrap();
        let mechanisms = self.mechanisms.read().unwrap();
        let _graph = self.graph.read().unwrap();

        // Check target exists
        if !nodes.contains_key(target) {
            return Err(format!("Target node '{}' not found", target));
        }

        // Create a copy of nodes for simulation
        let mut simulated = nodes.clone();

        // Apply intervention: set target value, remove incoming edges
        if let Some(node) = simulated.get_mut(target) {
            node.value = value.clone();
            node.confidence = 1.0; // Intervention sets with certainty
        }

        // Remove incoming edges to target (cut incoming edges)
        // We don't modify the actual graph, just don't use incoming edges for target
        // during forward propagation

        // Get descendants to propagate
        let descendants = self.get_descendants(target);

        // Topological sort of descendants
        let topo_order = self.topological_sort(&descendants)?;

        // Forward propagate through descendants
        for node_id in topo_order {
            if node_id == target {
                continue; // Already set by intervention
            }

            if let Some(mechanism) = mechanisms.get(&node_id) {
                // Get parent values
                let mut parent_values = HashMap::new();
                for parent in &mechanism.parents {
                    if let Some(parent_node) = simulated.get(parent) {
                        parent_values.insert(parent.clone(), parent_node.value.clone());
                    } else if let Some(ctx_val) = context.as_ref().and_then(|c| c.get(parent)) {
                        parent_values.insert(parent.clone(), ctx_val.clone());
                    } else {
                        // Parent not available - use mechanism default or skip
                        continue;
                    }
                }

                // Apply mechanism
                let outcome = self.apply_mechanism(mechanism, &parent_values)?;
                
                if let Some(node) = simulated.get_mut(&node_id) {
                    node.value = outcome;
                    node.confidence *= 0.95; // Confidence decays through propagation
                }
            }
        }

        // Build result
        let mut result = HashMap::new();
        for (id, node) in simulated {
            if descendants.contains(&id) || id == target {
                result.insert(id, node.value);
            }
        }

        // Log intervention
        self.log_intervention(target, value, &result, None);

        Ok(result)
    }

    /// Topological sort of a subgraph
    fn topological_sort(&self, nodes: &HashSet<String>) -> Result<Vec<String>, String> {
        let graph = self.graph.read().unwrap();
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize
        for node in nodes {
            in_degree.insert(node.clone(), 0);
            adj.insert(node.clone(), Vec::new());
        }

        // Build subgraph
        for node in nodes {
            if let Some(parents) = graph.get(node) {
                for parent in parents {
                    if nodes.contains(parent) {
                        adj.entry(parent.clone()).or_default().push(node.clone());
                        *in_degree.get_mut(node).unwrap() += 1;
                    }
                }
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<String> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(n, _)| n.clone())
            .collect();

        let mut result = Vec::new();
        while let Some(node) = queue.pop() {
            result.push(node.clone());
            for child in adj.get(&node).unwrap_or(&Vec::new()) {
                let deg = in_degree.get_mut(child).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(child.clone());
                }
            }
        }

        if result.len() != nodes.len() {
            return Err("Cycle detected in causal graph".to_string());
        }

        Ok(result)
    }

    /// Apply a causal mechanism to parent values
    fn apply_mechanism(
        &self,
        mechanism: &CausalMechanism,
        parent_values: &HashMap<String, NodeValue>,
    ) -> Result<NodeValue, String> {
        match mechanism.mech_type {
            MechanismType::Linear => {
                let mut sum = 0.0;
                for (parent, value) in parent_values {
                    let weight = mechanism.params.get(parent).unwrap_or(&1.0);
                    let val = match value {
                        NodeValue::Numeric(v) => *v,
                        NodeValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
                        _ => 0.0,
                    };
                    sum += weight * val;
                }
                // Add noise
                let noise = rand::random::<f32>() * mechanism.noise_var.sqrt();
                Ok(NodeValue::Numeric(sum + noise))
            },
            MechanismType::Threshold => {
                let mut sum = 0.0;
                for (parent, value) in parent_values {
                    let weight = mechanism.params.get(parent).unwrap_or(&1.0);
                    let val = match value {
                        NodeValue::Numeric(v) => *v,
                        NodeValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
                        _ => 0.0,
                    };
                    sum += weight * val;
                }
                let threshold = mechanism.params.get("threshold").unwrap_or(&0.5);
                Ok(NodeValue::Boolean(sum > *threshold))
            },
            MechanismType::Copy => {
                // Just copy the first parent
                if let Some((_, value)) = parent_values.iter().next() {
                    Ok(value.clone())
                } else {
                    Ok(NodeValue::Numeric(0.0))
                }
            },
            MechanismType::Categorical => {
                // Softmax over parents
                let mut max_val = f32::NEG_INFINITY;
                let mut max_key = String::new();
                for (parent, value) in parent_values {
                    let val = match value {
                        NodeValue::Numeric(v) => *v,
                        _ => 0.0,
                    };
                    if val > max_val {
                        max_val = val;
                        max_key = parent.clone();
                    }
                }
                Ok(NodeValue::Categorical(max_key))
            },
            MechanismType::Functional => {
                // Simplified: linear combination with learned weights
                let mut sum = 0.0;
                for (parent, value) in parent_values {
                    let weight = mechanism.params.get(parent).unwrap_or(&1.0);
                    let val = match value {
                        NodeValue::Numeric(v) => *v,
                        NodeValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
                        _ => 0.0,
                    };
                    sum += weight * val;
                }
                Ok(NodeValue::Numeric(sum.tanh())) // Non-linear activation
            },
        }
    }

    /// Counterfactual query: what would Y be if X = x, given observed evidence?
    pub fn counterfactual(
        &self,
        _target: &str,
        intervention: (&str, NodeValue),
        evidence: &HashMap<String, NodeValue>,
    ) -> Result<HashMap<String, NodeValue>, String> {
        // Abduction: update beliefs given evidence
        let updated_nodes = self.abduce(evidence)?;
        
        // Intervention
        self.do_intervention(intervention.0, intervention.1, Some(updated_nodes))
    }

    /// Abduction: update latent states given evidence
    fn abduce(&self, evidence: &HashMap<String, NodeValue>) -> Result<HashMap<String, NodeValue>, String> {
        // Simplified: just return evidence as updated beliefs
        // Full implementation would do Bayesian inference
        Ok(evidence.clone())
    }

    /// Log intervention for learning
    fn log_intervention(
        &self,
        target: &str,
        value: NodeValue,
        predicted: &HashMap<String, NodeValue>,
        actual: Option<HashMap<String, NodeValue>>,
    ) {
        let record = InterventionRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            target: target.to_string(),
            value,
            predicted_outcome: predicted.clone(),
            actual_outcome: actual.clone(),
            vfe_before: 0.0, // Would compute actual VFE
            vfe_after: actual.map(|_| 0.0),
        };
        self.intervention_log.write().unwrap().push(record);
    }

    /// Get VFE (Variational Free Energy) of current beliefs
    /// High VFE = high surprise = model needs updating
    pub fn compute_vfe(&self, beliefs: &HashMap<String, NodeValue>) -> f32 {
        let mut vfe = 0.0;
        let mechanisms = self.mechanisms.read().unwrap();

        for (node_id, _belief) in beliefs {
            if let Some(mechanism) = mechanisms.get(node_id) {
                // Surprisal = -log P(belief | mechanism)
                // Simplified: variance between belief and mechanism prediction
                // In practice, would compute proper KL divergence
                vfe += mechanism.noise_var;
            }
        }
        vfe
    }

    /// Learn from intervention outcome (update mechanisms)
    pub fn learn_from_intervention(&self, record: &InterventionRecord) {
        if let Some(actual) = &record.actual_outcome {
            let mut mechanisms = self.mechanisms.write().unwrap();
            
            for (node_id, actual_value) in actual {
                if let Some(mechanism) = mechanisms.get_mut(node_id) {
                    // Update mechanism parameters to reduce prediction error
                    // Simplified: gradient descent on mechanism parameters
                    let predicted = &record.predicted_outcome[node_id];
                    let error = self.value_difference(predicted, actual_value);
                    
                    // Adjust weights proportional to error
                    for (_parent, weight) in mechanism.params.iter_mut() {
                        *weight -= 0.01 * error; // Simple gradient step
                    }
                    mechanism.noise_var = (mechanism.noise_var + error.abs() * 0.01).max(0.001);
                }
            }
        }
    }

    fn value_difference(&self, a: &NodeValue, b: &NodeValue) -> f32 {
        match (a, b) {
            (NodeValue::Numeric(a), NodeValue::Numeric(b)) => (a - b).abs(),
            (NodeValue::Boolean(a), NodeValue::Boolean(b)) => if a == b { 0.0 } else { 1.0 },
            (NodeValue::Categorical(a), NodeValue::Categorical(b)) => if a == b { 0.0 } else { 1.0 },
            _ => 1.0,
        }
    }

    /// Query the model: what is the state of node?
    pub fn query(&self, node: &str) -> Option<NodeValue> {
        self.nodes.read().unwrap().get(node).map(|n| n.value.clone())
    }

    /// Get all node IDs
    pub fn get_all_nodes(&self) -> Vec<String> {
        self.nodes.read().unwrap().keys().cloned().collect()
    }
}

/// Mock Engram for testing
pub struct MockEngram;

impl EngramInterface for MockEngram {
    fn get_embedding(&self, _concept: &str) -> Option<Vec<f32>> {
        None
    }
    fn store_embedding(&self, _concept: &str, _embedding: Vec<f32>) {}
    fn query_similar(&self, _embedding: &[f32], _k: usize) -> Vec<(String, f32)> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scm_basic() {
        let engram = Arc::new(MockEngram);
        let scm = StructuralCausalModel::new(engram);

        // Add nodes
        scm.add_node(CausalNode {
            id: "rain".to_string(),
            node_type: NodeType::Observation,
            embedding: vec![0.0; 768],
            value: NodeValue::Boolean(false),
            confidence: 0.8,
            timestamp: 0,
        });

        scm.add_node(CausalNode {
            id: "wet_grass".to_string(),
            node_type: NodeType::State,
            embedding: vec![0.0; 768],
            value: NodeValue::Boolean(false),
            confidence: 0.5,
            timestamp: 0,
        });

        // Add mechanism: wet_grass = rain OR sprinkler
        scm.add_mechanism(CausalMechanism {
            child: "wet_grass".to_string(),
            parents: vec!["rain".to_string()],
            mech_type: MechanismType::Threshold,
            params: {
                let mut p = HashMap::new();
                p.insert("rain".to_string(), 1.0);
                p.insert("threshold".to_string(), 0.5);
                p
            },
            noise_var: 0.01,
        });

        // Test intervention
        let result = scm.do_intervention(
            "rain",
            NodeValue::Boolean(true),
            None,
        ).unwrap();

        assert_eq!(result.get("wet_grass"), Some(&NodeValue::Boolean(true)));
    }

    #[test]
    fn test_counterfactual() {
        let engram = Arc::new(MockEngram);
        let scm = StructuralCausalModel::new(engram);

        // Simple chain: X -> Y -> Z
        for id in ["x", "y", "z"] {
            scm.add_node(CausalNode {
                id: id.to_string(),
                node_type: NodeType::State,
                embedding: vec![0.0; 768],
                value: NodeValue::Numeric(0.0),
                confidence: 0.5,
                timestamp: 0,
            });
        }

        scm.add_mechanism(CausalMechanism {
            child: "y".to_string(),
            parents: vec!["x".to_string()],
            mech_type: MechanismType::Linear,
            params: { let mut p = HashMap::new(); p.insert("x".to_string(), 2.0); p },
            noise_var: 0.0,
        });

        scm.add_mechanism(CausalMechanism {
            child: "z".to_string(),
            parents: vec!["y".to_string()],
            mech_type: MechanismType::Linear,
            params: { let mut p = HashMap::new(); p.insert("y".to_string(), 3.0); p },
            noise_var: 0.0,
        });

        // Counterfactual: if x = 5, what is z? (given we observed y=10)
        let evidence: HashMap<String, NodeValue> = {
            let mut m = HashMap::new();
            m.insert("y".to_string(), NodeValue::Numeric(10.0));
            m
        };

        let result = scm.counterfactual(
            "z",
            ("x", NodeValue::Numeric(5.0)),
            &evidence,
        ).unwrap();

        // z = 3 * y = 3 * (2 * x) = 6 * x = 30
        assert_eq!(result.get("z"), Some(&NodeValue::Numeric(30.0)));
    }
}