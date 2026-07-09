use crate::model::EvalCase;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JudgePrompt {
    pub system: String,
    pub user: String,
}

impl JudgePrompt {
    pub fn build(case: &EvalCase, actual_output: &str) -> Self {
        Self {
            system: Self::system_prompt(),
            user: Self::user_prompt(case, actual_output),
        }
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

    fn user_prompt(case: &EvalCase, actual_output: &str) -> String {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_includes_case_fields_and_rubric() {
        let case = EvalCase::new("case-1", "trace-1", "What is 2+2?")
            .with_expected_output("4")
            .with_rubric("Check arithmetic.");

        let prompt = JudgePrompt::build(&case, "Four.");

        assert!(prompt.system.contains("evaluation judge"));
        assert!(prompt.user.contains("case-1"));
        assert!(prompt.user.contains("trace-1"));
        assert!(prompt.user.contains("What is 2+2?"));
        assert!(prompt.user.contains("Four."));
        assert!(prompt.user.contains("Check arithmetic."));
    }
}
