use anyhow::Result;

#[cfg(any(
    feature = "llm-judge-openai",
    feature = "embeddings-openai",
    feature = "cluster-label-openai"
))]
#[tokio::main]
async fn main() -> Result<()> {
    traces_to_evals::cli::run().await
}

#[cfg(not(any(
    feature = "llm-judge-openai",
    feature = "embeddings-openai",
    feature = "cluster-label-openai"
)))]
fn main() -> Result<()> {
    traces_to_evals::cli::run()
}
