use serde::Deserialize;
use traces_to_evals::{AgentContextReleaseV1, AgentTaxonomyReleaseV1, EvaluatorReleaseSpecV1};

#[derive(Deserialize)]
struct ExpectedIds {
    evaluator_release_id: String,
    agent_context_release_id: String,
    agent_taxonomy_release_id: String,
}

#[test]
fn learned_contract_fixtures_validate_and_keep_golden_identities() {
    let evaluator: EvaluatorReleaseSpecV1 = serde_json::from_str(include_str!(
        "../fixtures/learned/evaluator_release_prompt_judge.json"
    ))
    .unwrap();
    let context: AgentContextReleaseV1 = serde_json::from_str(include_str!(
        "../fixtures/learned/agent_context_release.json"
    ))
    .unwrap();
    let taxonomy: AgentTaxonomyReleaseV1 = serde_json::from_str(include_str!(
        "../fixtures/learned/agent_taxonomy_release.json"
    ))
    .unwrap();
    let expected: ExpectedIds =
        serde_json::from_str(include_str!("../fixtures/learned/expected_ids.json")).unwrap();

    assert_eq!(
        evaluator.release_id().unwrap(),
        expected.evaluator_release_id
    );
    assert_eq!(
        context.release_id().unwrap(),
        expected.agent_context_release_id
    );
    assert_eq!(
        taxonomy.release_id().unwrap(),
        expected.agent_taxonomy_release_id
    );
}
