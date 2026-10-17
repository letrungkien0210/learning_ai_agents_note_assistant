from collections.abc import Mapping

API_KEY_ENV_VAR = "GOOGLE_API_KEY"
MODEL_NAME_ENV_VAR = "GEMINI_MODEL"
DEFAULT_MODEL_NAME = "gemini-3.8-flash"


class MissingApiKeyError(Exception):
    pass


def read_api_key(environ: Mapping[str, str]) -> str:
    api_key = environ.get(API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise MissingApiKeyError(f"Environment variable {API_KEY_ENV_VAR} is missing or blank.")
    return api_key


def read_model_name(environ: Mapping[str, str]) -> str:
    model_name = environ.get(MODEL_NAME_ENV_VAR, "").strip()
    return model_name or DEFAULT_MODEL_NAME
