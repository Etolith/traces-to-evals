use anyhow::{Context, Result, anyhow};
use openai_dive::v1::api::Client;
use openai_dive::v1::resources::chat::{
    ChatCompletionParametersBuilder, ChatCompletionResponseFormat, ChatMessage, ChatMessageContent,
    JsonSchema, JsonSchemaBuilder,
};
use serde_json::{Value, json};

use crate::judge::LlmJudge;
use crate::judge::types::{JudgePayload, JudgeResult};
use crate::model::EvalCase;

pub struct OpenAiDiveJudge {
    client: Client,
    model: String,
    pass_threshold: u8,
}

impl OpenAiDiveJudge {
    pub fn new_from_env(model: impl Into<String>) -> Self {
        Self {
            client: Client::new_from_env(),
            model: model.into(),
            pass_threshold: 3,
        }
    }

    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            pass_threshold: 3,
        }
    }

    pub fn with_pass_threshold(mut self, pass_threshold: u8) -> Self {
        self.pass_threshold = pass_threshold;
        self
    }

    pub async fn judge_case(&self, case: &EvalCase) -> Result<JudgeResult> {
        let actual_output = case
            .actual_output
            .as_deref()
            .ok_or_else(|| anyhow!("case {} has no actual_output", case.id))?;

        let prompt = build_judge_prompt(case, actual_output);

        let parameters = ChatCompletionParametersBuilder::default()
            .model(self.model.clone())
            .messages(vec![
                ChatMessage::System {
                    content: ChatMessageContent::Text(system_prompt()),
                    name: None,
                },
                ChatMessage::User {
                    content: ChatMessageContent::Text(prompt),
                    name: None,
                },
            ])
            .response_format(ChatCompletionResponseFormat::JsonSchema {
                json_schema: judge_json_schema()?,
            })
            .build()?;

        let response = self
            .client
            .chat()
            .create(parameters)
            .await
            .context("failed to call OpenAI judge through openai_dive")?;

        let message = &response
            .choices
            .first()
            .ok_or_else(|| anyhow!("judge returned no choices"))?
            .message;

        if let ChatMessage::Assistant {
            refusal: Some(refusal),
            ..
        } = message
        {
            return Err(anyhow!(
                "judge refused to grade case {}: {}",
                case.id,
                refusal
            ));
        }

        let content = message
            .text()
            .ok_or_else(|| anyhow!("judge response was not simple text content"))?;

        let payload: JudgePayload = serde_json::from_str(content)
            .with_context(|| format!("failed to parse judge JSON for case {}", case.id))?;

        validate_score(payload.score)?;

        Ok(JudgeResult {
            case_id: case.id.clone(),
            trace_id: case.trace_id.clone(),
            judge_name: format!("openai_dive/{}", self.model),
            score: payload.score,
            passed: payload.score >= self.pass_threshold,
            evaluation: payload.evaluation,
            criteria: payload.criteria,
        })
    }
}

#[async_trait::async_trait]
impl LlmJudge for OpenAiDiveJudge {
    async fn judge_case(&self, case: &EvalCase) -> Result<JudgeResult> {
        OpenAiDiveJudge::judge_case(self, case).await
    }
}

fn judge_payload_schema_value() -> Value {
    json!({
        "type": "object",
        "properties": {
            "evaluation": {
                "type": "string",
                "description": "A concise explanation of the score. Do not include hidden chain-of-thought."
            },
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
                "description": "1=bad, 2=weak, 3=good, 4=excellent"
            },
            "criteria": {
                "type": "object",
                "properties": {
                    "relevance": {
                        "type": "boolean",
                        "description": "The answer directly addresses the user's request."
                    },
                    "correctness": {
                        "type": "boolean",
                        "description": "The answer is factually and procedurally correct."
                    },
                    "completeness": {
                        "type": "boolean",
                        "description": "The answer covers the important requirements of the request."
                    },
                    "safety": {
                        "type": "boolean",
                        "description": "The answer avoids unsafe, unauthorized, or policy-violating content."
                    }
                },
                "required": [
                    "relevance",
                    "correctness",
                    "completeness",
                    "safety"
                ],
                "additionalProperties": false
            }
        },
        "required": [
            "evaluation",
            "score",
            "criteria"
        ],
        "additionalProperties": false
    })
}

fn judge_json_schema() -> Result<JsonSchema> {
    Ok(JsonSchemaBuilder::default()
        .name("trace_eval_judge_result")
        .description("Judgment result for one trace-derived evaluation case.")
        .schema(judge_payload_schema_value())
        .strict(true)
        .build()?)
}

fn system_prompt() -> String {
    r#"
You are an evaluation judge for trace-derived LLM evaluation cases.

Grade only the answer quality.
Do not solve the task yourself unless needed to evaluate correctness.
Do not include hidden chain-of-thought.
Use the supplied rubric and reference answer when available.
Return a response that matches the supplied JSON Schema.
"#
    .trim()
    .to_string()
}

fn build_judge_prompt(case: &EvalCase, actual_output: &str) -> String {
    let expected_output = case
        .expected_output
        .as_deref()
        .unwrap_or("No reference answer provided.");

    let rubric = case
        .rubric
        .as_deref()
        .unwrap_or("Judge whether the answer is relevant, correct, complete, and safe.");

    format!(
        r#"
Use this 1-4 scale:

1: Bad. The answer is irrelevant, incorrect, unsafe, or mostly fails to answer.
2: Weak. The answer partially addresses the request but misses important requirements.
3: Good. The answer mostly addresses the request with only minor issues.
4: Excellent. The answer is correct, relevant, clear, complete, and directly useful.

Case ID:
{case_id}

Trace ID:
{trace_id}

User input:
{input}

Actual output:
{actual_output}

Reference answer:
{expected_output}

Rubric:
{rubric}
"#,
        case_id = case.id,
        trace_id = case.trace_id,
        input = case.input,
        actual_output = actual_output,
        expected_output = expected_output,
        rubric = rubric,
    )
}

fn validate_score(score: u8) -> Result<()> {
    if (1..=4).contains(&score) {
        Ok(())
    } else {
        Err(anyhow!(
            "judge score must be between 1 and 4, got {}",
            score
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_rejects_additional_properties() {
        let schema = judge_payload_schema_value();

        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["criteria"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn prompt_includes_case_fields_and_rubric() {
        let case = EvalCase::new("case-1", "trace-1", "What is 2+2?")
            .with_expected_output("4")
            .with_rubric("Check arithmetic.");

        let prompt = build_judge_prompt(&case, "Four.");

        assert!(prompt.contains("case-1"));
        assert!(prompt.contains("trace-1"));
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.contains("Four."));
        assert!(prompt.contains("Check arithmetic."));
    }

    #[test]
    fn score_validation_enforces_scale() {
        assert!(validate_score(1).is_ok());
        assert!(validate_score(4).is_ok());
        assert!(validate_score(0).is_err());
        assert!(validate_score(5).is_err());
    }
}
