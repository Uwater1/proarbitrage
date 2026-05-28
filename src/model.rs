use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;
use anyhow::{Result, Context};

#[derive(Deserialize, Debug, Clone)]
pub struct XGBTree {
    pub left_children: Vec<i32>,
    pub right_children: Vec<i32>,
    pub split_indices: Vec<i32>,
    pub split_conditions: Vec<f64>,
    pub base_weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct XGBModel {
    pub trees: Vec<XGBTree>,
    pub base_score: f64,
}

impl XGBModel {
    pub fn load_from_json(path: &str) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("Failed to open model file: {}", path))?;
        let reader = BufReader::new(file);
        let val: serde_json::Value = serde_json::from_reader(reader)
            .with_context(|| "Failed to parse JSON from model file")?;

        let learner = val.get("learner").context("No 'learner' field in model JSON")?;
        
        // Extract base_score from learner_model_param
        let base_score_str = learner
            .get("learner_model_param")
            .and_then(|p| p.get("base_score"))
            .and_then(|s| s.as_str())
            .context("No 'base_score' string found in 'learner_model_param'")?;
        
        // Clean bracket representation like "[-2.4777513E-3]"
        let trimmed = base_score_str
            .trim_start_matches('[')
            .trim_end_matches(']');
        
        let base_score = trimmed.parse::<f64>()
            .with_context(|| format!("Failed to parse base_score: {}", base_score_str))?;

        // Extract trees array
        let trees_val = learner
            .get("gradient_booster")
            .and_then(|gb| gb.get("model"))
            .and_then(|m| m.get("trees"))
            .context("No 'trees' array found in model JSON")?;
        
        let trees: Vec<XGBTree> = serde_json::from_value(trees_val.clone())
            .with_context(|| "Failed to deserialize trees array")?;

        Ok(Self { trees, base_score })
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut prediction = self.base_score;
        for tree in &self.trees {
            let mut node = 0usize;
            while tree.left_children[node] != -1 {
                let feat_idx = tree.split_indices[node] as usize;
                let val = features[feat_idx];
                let threshold = tree.split_conditions[node];
                if val < threshold {
                    node = tree.left_children[node] as usize;
                } else {
                    node = tree.right_children[node] as usize;
                }
            }
            prediction += tree.base_weights[node];
        }
        prediction
    }
}
