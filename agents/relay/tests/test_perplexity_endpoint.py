import unittest
from importlib.metadata import version
from unittest.mock import AsyncMock, Mock

from relay.providers.perplexity.endpoint import PerplexityAgentAPIEndpoint


API_KEY = "test-api-key"
PROMPT = "test prompt"
INTEGRATION = f"wandr/{version('relay')}"


class PerplexityEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_response_sends_attribution_headers(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"id": "response-id"}
        client = Mock()
        client.post = AsyncMock(return_value=response)
        endpoint = PerplexityAgentAPIEndpoint(api_key=API_KEY)

        result = await endpoint._create_response(client, PROMPT)

        self.assertEqual(result, {"id": "response-id"})
        client.post.assert_awaited_once_with(
            "https://api.perplexity.ai/v2/responses",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": INTEGRATION,
                "X-Pplx-Integration": INTEGRATION,
            },
            json={
                "model": "openai/gpt-5.5",
                "input": PROMPT,
                "tools": [{"type": "web_search"}, {"type": "sandbox"}],
                "max_steps": 100,
                "max_output_tokens": 128000,
                "reasoning": {"effort": "high"},
                "background": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
