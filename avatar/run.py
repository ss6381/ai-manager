import logging
import os

import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from pydantic import BaseModel

import outspeed as sp
import json

nest_asyncio.apply()

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_outspeed_version():
    import importlib.metadata

    from packaging import version

    required_version = "0.1.151"

    try:
        current_version = importlib.metadata.version("outspeed")
        if version.parse(current_version) < version.parse(required_version):
            raise ValueError(f"Outspeed version {current_version} is not greater than {required_version}.")
        else:
            print(f"Outspeed version {current_version} meets the requirement.")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Outspeed package is not installed.")


check_outspeed_version()


class Query(BaseModel):
    query_for_neural_search: str


class SearchResult(BaseModel):
    result: str


class SearchTool(sp.Tool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters_type: type[Query],
        response_type: type[SearchResult],
        query_engine,
    ):
        super().__init__(name, description, parameters_type, response_type)
        self.query_engine = query_engine

    async def run(self, query: Query) -> SearchResult:
        logging.info(f"Searching for: {query.query_for_neural_search}")
        response = self.query_engine.query(query.query_for_neural_search)
        logging.info(f"RAG Response: {response}")
        return SearchResult(result=str(response))

@sp.App()
class VoiceBot:
    async def setup(self) -> None:
        # Initialize the AI services
        self.deepgram_node = sp.DeepgramSTT(sample_rate=8000)
        self.llm_node = sp.GroqLLM(
            system_prompt="You are a helpful assistant. Keep your answers very short. No special characters in responses.",
        )
        self.token_aggregator_node = sp.TokenAggregator()
        self.tts_node = sp.CartesiaTTS(
            voice_id="95856005-0332-41b0-935f-352e296aa0df",
            api_key=os.getenv("CARTESIA_API_KEY")  # Ensure API key is loaded from environment
        )

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream) -> sp.AudioStream:
        deepgram_stream: sp.TextStream = self.deepgram_node.run(audio_input_queue)

        text_input_queue = sp.map(text_input_queue, lambda x: json.loads(x).get("content"))

        llm_input_queue: sp.TextStream = sp.merge(
            [deepgram_stream, text_input_queue],
        )

        llm_token_stream, chat_history_stream = self.llm_node.run(llm_input_queue)

        token_aggregator_stream: sp.TextStream = self.token_aggregator_node.run(llm_token_stream)
        tts_stream: sp.AudioStream = self.tts_node.run(token_aggregator_stream)

        return tts_stream

    async def teardown(self) -> None:
        await self.deepgram_node.close()
        await self.llm_node.close()
        await self.token_aggregator_node.close()
        await self.tts_node.close()

if __name__ == "__main__":
    VoiceBot().start()