from google.genai.errors import APIError

HTTP_PAYMENT_REQUIRED = 402
HTTP_TOO_MANY_REQUESTS = 429
QUOTA_ERROR_CODES = frozenset({HTTP_PAYMENT_REQUIRED, HTTP_TOO_MANY_REQUESTS})
RESOURCE_EXHAUSTED_STATUS = "RESOURCE_EXHAUSTED"


def _is_single_quota_error(error: BaseException) -> bool:
    if not isinstance(error, APIError):
        return False
    return (
        error.code in QUOTA_ERROR_CODES
        or error.status == RESOURCE_EXHAUSTED_STATUS
    )


def is_quota_exhausted(error: BaseException) -> bool:
    seen_ids: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen_ids:
        if _is_single_quota_error(current):
            return True
        seen_ids.add(id(current))
        current = current.__cause__
    return False


def describe_quota_error(model_name: str) -> str:
    return (
        f"Error: Gemini API quota or prepaid credits are exhausted for model '{model_name}'.\n"
        "The project associated with your current GOOGLE_API_KEY has no remaining quota or credits.\n"
        "To resolve this:\n"
        "  1. Top up prepaid credits or check project billing at https://ai.studio/projects\n"
        "  2. Or set GOOGLE_API_KEY to an API key from a project with available quota.\n"
        "Learn more about billing: https://ai.google.dev/gemini-api/docs/billing#prepay"
    )
