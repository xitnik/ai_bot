from types import SimpleNamespace
from typing import Optional

import llm_client


class DummyChatCompletions:
    def __init__(self, parent: "DummyOpenAI") -> None:
        self.parent = parent
        self.last_kwargs: Optional[dict] = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), text=None)]
        )


class DummyChat:
    def __init__(self, parent: "DummyOpenAI") -> None:
        self.parent = parent
        self.completions = DummyChatCompletions(parent)


class DummyEmbeddings:
    def __init__(self, parent: "DummyOpenAI") -> None:
        self.parent = parent
        self.last: Optional[tuple] = None

    def create(self, model, input):  # type: ignore[override]
        self.last = (model, input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0])])


class DummyOpenAI:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = DummyChat(self)
        self.embeddings = DummyEmbeddings(self)


def test_get_client_and_chat(monkeypatch):
    # Подменяем окружение и SDK, чтобы не ходить в сеть.
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "dummy")
    monkeypatch.setattr(llm_client, "_client", None)
    monkeypatch.setattr(llm_client, "OpenAI", DummyOpenAI)

    client = llm_client.get_client()
    assert isinstance(client, DummyOpenAI)
    assert client.base_url == "http://localhost:4000"
    assert client.api_key == "dummy"

    response = llm_client.chat(model="test-model", messages=[{"role": "user", "content": "ping"}])
    assert response.choices[0].message.content == "ok"
    assert client.chat.completions.last_kwargs is not None
    assert client.chat.completions.last_kwargs["model"] == "test-model"
    assert client.chat.completions.last_kwargs["messages"][0]["content"] == "ping"

    vectors = llm_client.embeddings(model="embed-model", inputs=["hi"])
    assert vectors == [[1.0, 2.0]]
    assert client.embeddings.last == ("embed-model", ["hi"])
