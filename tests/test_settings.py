import pytest

from settings import (
    DEFAULT_MODEL_NAME,
    MissingApiKeyError,
    read_api_key,
    read_model_name,
)


def test_read_api_key_returns_value_when_present():
    assert read_api_key({"GOOGLE_API_KEY": "abc"}) == "abc"


def test_read_api_key_strips_surrounding_whitespace():
    assert read_api_key({"GOOGLE_API_KEY": "  abc \n"}) == "abc"


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"GOOGLE_API_KEY": ""},
        {"GOOGLE_API_KEY": "   "},
    ],
)
def test_read_api_key_raises_when_missing_or_blank(environ):
    with pytest.raises(MissingApiKeyError):
        read_api_key(environ)


def test_read_model_name_returns_configured_model():
    assert read_model_name({"GEMINI_MODEL": "gemini-x"}) == "gemini-x"


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"GEMINI_MODEL": ""},
        {"GEMINI_MODEL": "   "},
    ],
)
def test_read_model_name_falls_back_to_default_when_missing_or_blank(environ):
    assert read_model_name(environ) == DEFAULT_MODEL_NAME
    assert DEFAULT_MODEL_NAME == "gemini-3.8-flash"
