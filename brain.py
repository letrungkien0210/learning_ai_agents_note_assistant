from collections.abc import Mapping
import os
import sys

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from quota_errors import describe_quota_error, is_quota_exhausted
from settings import MissingApiKeyError, read_api_key, read_model_name
from tools import them_ghi_chu, doc_ghi_chu, kiem_duyet_cong_viec

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
    danh_sach_cong_cu = [them_ghi_chu, doc_ghi_chu, kiem_duyet_cong_viec]
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
    tin_nhan = [HumanMessage(content="Hôm nay tôi có việc gì cần làm không?")]

    # 4. Yêu cầu Agent xử lý tin nhắn
    try:
        phan_hoi = llm_kem_cong_cu.invoke(tin_nhan)
    except ChatGoogleGenerativeAIError as error:
        if is_quota_exhausted(error):
            print(describe_quota_error(model_name), file=sys.stderr)
            return EXIT_QUOTA_EXHAUSTED
        raise

    # Nhớ thêm danh bạ này vào trước vòng lặp nhé:
    danh_ba_cong_cu = {
        "them_ghi_chu": them_ghi_chu,
        "doc_ghi_chu": doc_ghi_chu,
        "kiem_duyet_cong_viec": kiem_duyet_cong_viec
    }

    # Tạo trí nhớ (danh sách rỗng) TRƯỚC khi bắt đầu vòng lặp
    tin_nhan = []

    # Giao việc cho Sếp (System Prompt)
    huong_dan_cho_sep = """Bạn là một Quản lý ghi chú (Router Agent). 
    Quy trình làm việc BẮT BUỘC của bạn khi người dùng yêu cầu thêm ghi chú:
    1. Gọi 'kiem_duyet_cong_viec' để lấy nhãn phân loại (Khẩn cấp/Bình thường).
    2. Gắn nhãn đó vào đầu nội dung công việc (Ví dụ: [Khẩn cấp] 🚨 Chiều nay đi siêu thị).
    3. Gọi 'them_ghi_chu' để lưu nội dung đã có nhãn vào sổ.
    Tuyệt đối không được tự ý ghi vào sổ mà chưa qua bước kiểm duyệt."""
    tin_nhan.append(SystemMessage(content=huong_dan_cho_sep))
    
    print("Trợ lý AI đã sẵn sàng! 🤖 (Gõ 'Exit' để kết thúc)")

    # Mở vòng lặp vô hạn
    while True:
        cau_hoi = input("\nBạn: ")
    
        if cau_hoi.lower() == 'Exit' or cau_hoi.lower() == 'exit':
            print("Tạm biệt!")
            break

        tin_nhan.append(HumanMessage(content=cau_hoi))

        # [ReAct - Reason]: AI phân tích ngữ cảnh và quyết định hành động (Action/Tool Calls)
        phan_hoi = llm_kem_cong_cu.invoke(tin_nhan)

        # Vòng lặp suy luận: Chừng nào AI còn yêu cầu gọi công cụ, thì tiếp tục xử lý
        while phan_hoi.tool_calls:
            # [ReAct - Act]: Agent thực thi công cụ (Tool Execution)
            for yeu_cau in phan_hoi.tool_calls:
                ten_cong_cu = yeu_cau['name']
                # Tra cứu trong danh bạ để lấy hàm công cụ ra
                cong_cu_duoc_chon = danh_ba_cong_cu[ten_cong_cu];

                # Chạy công cụ
                ket_qua = cong_cu_duoc_chon.invoke(yeu_cau['args']);
                print(f"Hệ thống báo: {ket_qua}")

                # [ReAct - Reason]: Hoàn thiện vòng lặp React
                # Nối quyết định gọi công cụ và kết quả vào lịch sử tin nhắn
                tin_nhan.append(phan_hoi) 
                tin_nhan.append(ToolMessage(content=ket_qua, tool_call_id=yeu_cau['id']))
                    
            # Gọi lại AI để xem nó muốn làm gì tiếp theo sau khi có kết quả công cụ
            phan_hoi = llm_kem_cong_cu.invoke(tin_nhan)
        
        # 6. Khi AI không gọi công cụ nữa, in ra câu trả lời cuối cùng và lưu vào bộ nhớ
        noi_dung_tra_loi = phan_hoi.content

        # Mẹo bóc vỏ: Nếu nội dung bị gói trong danh sách (list), ta rút phần 'text' ra
        if isinstance(noi_dung_tra_loi, list):
            noi_dung_tra_loi = noi_dung_tra_loi[0]['text']
        
        print("Trợ lý AI:", noi_dung_tra_loi)
        tin_nhan.append(phan_hoi)

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())