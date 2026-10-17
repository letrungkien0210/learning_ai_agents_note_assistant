from google.genai.errors import ClientError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from quota_errors import describe_quota_error, is_quota_exhausted


def _make_client_error(code: int, status: str, message: str = "error") -> ClientError:
    return ClientError(
        code,
        {"error": {"code": code, "message": message, "status": status}},
    )


def _wrap_in_langchain_error(cause: BaseException) -> ChatGoogleGenerativeAIError:
    try:
        raise ChatGoogleGenerativeAIError(f"Error calling model: {cause}") from cause
    except ChatGoogleGenerativeAIError as wrapped:
        return wrapped


def test_is_quota_exhausted_true_for_chained_402_resource_exhausted():
    cause = _make_client_error(
        402,
        "RESOURCE_EXHAUSTED",
        "Your prepayment credits are depleted.",
    )
    error = _wrap_in_langchain_error(cause)
    assert is_quota_exhausted(error) is True


def test_is_quota_exhausted_true_for_chained_429_error():
    cause = _make_client_error(429, "TOO_MANY_REQUESTS", "Quota exceeded.")
    error = _wrap_in_langchain_error(cause)
    assert is_quota_exhausted(error) is True


def test_is_quota_exhausted_true_for_bare_client_error_402():
    error = _make_client_error(402, "RESOURCE_EXHAUSTED")
    assert is_quota_exhausted(error) is True


def test_is_quota_exhausted_false_for_chained_400_invalid_argument():
    cause = _make_client_error(400, "INVALID_ARGUMENT", "Invalid API key.")
    error = _wrap_in_langchain_error(cause)
    assert is_quota_exhausted(error) is False


def test_is_quota_exhausted_false_for_plain_value_error_with_matching_text():
    error = ValueError("402 RESOURCE_EXHAUSTED")
    assert is_quota_exhausted(error) is False


def test_describe_quota_error_contains_model_and_remediation_links():
    message = describe_quota_error("gemini-x")
    assert "gemini-x" in message
    assert "https://ai.studio/projects" in message
    assert "GOOGLE_API_KEY" in message
    assert "https://ai.google.dev/gemini-api/docs/billing#prepay" in message
