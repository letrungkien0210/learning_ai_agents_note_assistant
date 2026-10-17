from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import brain
from brain import EXIT_MISSING_API_KEY, EXIT_QUOTA_EXHAUSTED, main


class _FakeAgent:
    def __init__(self, response=None, error: BaseException | None = None):
        self._response = response
        self._error = error
        self.invocations: list[object] = []

    def invoke(self, messages):
        self.invocations.append(messages)
        if self._error is not None:
            raise self._error
        return self._response


def _make_chained_llm_error(code: int, status: str, message: str) -> ChatGoogleGenerativeAIError:
    cause = ClientError(
        code,
        {"error": {"code": code, "message": message, "status": status}},
    )
    try:
        raise ChatGoogleGenerativeAIError(f"Error calling model ({status}): {code}") from cause
    except ChatGoogleGenerativeAIError as wrapped:
        return wrapped


def test_main_returns_missing_api_key_exit_code_when_key_absent(monkeypatch, capsys):
    built_calls: list[tuple[str, str]] = []

    def fake_build_agent(api_key: str, model_name: str):
        built_calls.append((api_key, model_name))
        return _FakeAgent()

    monkeypatch.setattr(brain, "build_agent", fake_build_agent)

    exit_code = main({})

    captured = capsys.readouterr()
    assert exit_code == EXIT_MISSING_API_KEY == 2
    assert "GOOGLE_API_KEY" in captured.err
    assert captured.out == ""
    assert built_calls == []


def test_main_prints_tool_calls_and_returns_zero_on_success(monkeypatch, capsys):
    fake_response = SimpleNamespace(
        tool_calls=[
            {
                "name": "them_ghi_chu",
                "args": {"noi_dung": "chiều nay đi siêu thị mua rau"},
                "id": "call_1",
            }
        ]
    )
    fake_agent = _FakeAgent(response=fake_response)
    built_with: list[tuple[str, str]] = []

    def fake_build_agent(api_key: str, model_name: str):
        built_with.append((api_key, model_name))
        return fake_agent

    monkeypatch.setattr(brain, "build_agent", fake_build_agent)

    exit_code = main({"GOOGLE_API_KEY": "k"})

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "them_ghi_chu" in captured.out
    assert captured.err == ""
    assert built_with == [("k", "gemini-3.8-flash")]
    assert len(fake_agent.invocations) == 1


def test_main_prints_quota_message_and_returns_exit_code_3_on_402_quota_error(
    monkeypatch, capsys
):
    quota_error = _make_chained_llm_error(
        402,
        "RESOURCE_EXHAUSTED",
        "Your prepayment credits are depleted.",
    )
    fake_agent = _FakeAgent(error=quota_error)

    monkeypatch.setattr(
        brain,
        "build_agent",
        lambda api_key, model_name: fake_agent,
    )

    exit_code = main({"GOOGLE_API_KEY": "k", "GEMINI_MODEL": "gemini-custom-test"})

    captured = capsys.readouterr()
    assert exit_code == EXIT_QUOTA_EXHAUSTED == 3
    assert "gemini-custom-test" in captured.err
    assert "ai.studio" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_main_reraises_non_quota_langchain_error(monkeypatch):
    non_quota_error = _make_chained_llm_error(
        400,
        "INVALID_ARGUMENT",
        "API key not valid.",
    )
    fake_agent = _FakeAgent(error=non_quota_error)

    monkeypatch.setattr(
        brain,
        "build_agent",
        lambda api_key, model_name: fake_agent,
    )

    with pytest.raises(ChatGoogleGenerativeAIError) as exc_info:
        main({"GOOGLE_API_KEY": "k"})

    assert exc_info.value is non_quota_error
