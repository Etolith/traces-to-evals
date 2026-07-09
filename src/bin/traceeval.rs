use anyhow::Result;

#[cfg(feature = "llm-judge-openai")]
#[tokio::main]
async fn main() -> Result<()> {
    traces_to_evals::cli::run().await
}

#[cfg(not(feature = "llm-judge-openai"))]
fn main() -> Result<()> {
    traces_to_evals::cli::run()
}
