use openai_dive::v1::api::Client;

use crate::providers::openai_dive::chat::OpenAiChatClient;

pub struct OpenAiClient {
    chat: OpenAiChatClient,
}

impl OpenAiClient {
    pub fn from_env() -> Self {
        Self {
            chat: OpenAiChatClient::from_env(),
        }
    }

    pub fn new(client: Client) -> Self {
        Self {
            chat: OpenAiChatClient::new(client),
        }
    }

    pub fn chat(&self) -> &OpenAiChatClient {
        &self.chat
    }
}
