from collections.abc import Mapping
import os
import sys

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from quota_errors import describe_quota_error, is_quota_exhausted
from settings import MissingApiKeyError, read_api_key, read_model_name
from tools import them_ghi_chu

EXIT_SUCCESS = 0
EXIT_MISSING_API_KEY = 2
EXIT_QUOTA_EXHAUSTED = 3

MISSING_API_KEY_HINT = (
    "Error: GOOGLE_API_KEY is not set.\n"
    'Set it in PowerShell before running: $env:GOOGLE_API_KEY = "your_gemini_api_key_here"'
)


def build_agent(api_key: str, model_name: str):
    # 1. Khởi tạo "bộ não" Gemini
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    # 2. Kết nối công cụ với mô hình AI
    danh_sach_cong_cu = [them_ghi_chu]
    llm_kem_cong_cu = llm.bind_tools(danh_sach_cong_cu)
    return llm_kem_cong_cu


def main(environ: Mapping[str, str] = os.environ) -> int:
    try:
        api_key = read_api_key(environ)
    except MissingApiKeyError:
        print(MISSING_API_KEY_HINT, file=sys.stderr)
        return EXIT_MISSING_API_KEY

    model_name = read_model_name(environ)
    llm_kem_cong_cu = build_agent(api_key=api_key, model_name=model_name)

    # 3. Tạo tin nhắn yêu cầu từ người dùng
    tin_nhan = [HumanMessage(content="Nhắc tôi chiều nay đi siêu thị mua rau")]

    # 4. Yêu cầu Agent xử lý tin nhắn
    try:
        phan_hoi = llm_kem_cong_cu.invoke(tin_nhan)
    except ChatGoogleGenerativeAIError as error:
        if is_quota_exhausted(error):
            print(describe_quota_error(model_name), file=sys.stderr)
            return EXIT_QUOTA_EXHAUSTED
        raise

    # In quyết định gọi công cụ của Agent ra màn hình
    print(phan_hoi.tool_calls)
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())