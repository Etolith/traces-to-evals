use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeResult {
    pub case_id: String,
    pub trace_id: String,
    pub judge_name: String,
    pub score: u8,
    pub passed: bool,
    pub evaluation: String,
    pub criteria: JudgeCriteria,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgePayload {
    pub evaluation: String,
    pub score: u8,
    pub criteria: JudgeCriteria,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeCriteria {
    pub relevance: bool,
    pub correctness: bool,
    pub completeness: bool,
    pub safety: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_payload_fields() {
        let json = r#"{
            "evaluation": "fine",
            "score": 3,
            "passed": true,
            "criteria": {
                "relevance": true,
                "correctness": true,
                "completeness": true,
                "safety": true
            }
        }"#;

        assert!(serde_json::from_str::<JudgePayload>(json).is_err());
    }
}
