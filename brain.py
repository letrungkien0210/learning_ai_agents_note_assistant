import os
import sys
from collections.abc import Mapping

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


class TroLyGhiChu:
    """Lớp đóng gói toàn bộ logic của Multi-Agent Note Assistant."""
    
    def __init__(self, api_key: str, model_name: str):
        # 1. Khởi tạo LLM và gắn công cụ
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        
        danh_sach_cong_cu = [them_ghi_chu, doc_ghi_chu, kiem_duyet_cong_viec]
        self.llm_kem_cong_cu = self.llm.bind_tools(danh_sach_cong_cu)
        
        # Danh bạ tra cứu công cụ
        self.danh_ba_cong_cu = {
            "them_ghi_chu": them_ghi_chu,
            "doc_ghi_chu": doc_ghi_chu,
            "kiem_duyet_cong_viec": kiem_duyet_cong_viec
        }
        
        # 2. Khởi tạo trí nhớ và System Prompt (Chỉ thị cho Sếp)
        self.tin_nhan = []
        huong_dan_cho_sep = """Bạn là một Quản lý ghi chú (Router Agent). 
        Quy trình làm việc BẮT BUỘC của bạn khi người dùng yêu cầu thêm ghi chú:
        1. Gọi 'kiem_duyet_cong_viec' để lấy nhãn phân loại (Khẩn cấp/Bình thường).
        2. Gắn nhãn đó vào đầu nội dung công việc (Ví dụ: [Khẩn cấp] 🚨 Chiều nay đi siêu thị).
        3. Gọi 'them_ghi_chu' để lưu nội dung đã có nhãn vào sổ.
        Tuyệt đối không được tự ý ghi vào sổ mà chưa qua bước kiểm duyệt."""
        
        self.tin_nhan.append(SystemMessage(content=huong_dan_cho_sep))

    def xu_ly_tin_nhan(self, cau_hoi: str) -> str:
        """Xử lý vòng lặp suy luận ReAct cho một lượt chat."""
        self.tin_nhan.append(HumanMessage(content=cau_hoi))
        phan_hoi = self.llm_kem_cong_cu.invoke(self.tin_nhan)

        # Vòng lặp suy luận đa bước
        while phan_hoi.tool_calls:
            for yeu_cau in phan_hoi.tool_calls:
                ten_cong_cu = yeu_cau['name']
                cong_cu_duoc_chon = self.danh_ba_cong_cu[ten_cong_cu]
                
                # Thực thi
                ket_qua = cong_cu_duoc_chon.invoke(yeu_cau['args'])
                print(f"Hệ thống báo ({ten_cong_cu}): {ket_qua}")
                
                # Lưu lịch sử
                self.tin_nhan.append(phan_hoi) 
                self.tin_nhan.append(ToolMessage(content=ket_qua, tool_call_id=yeu_cau['id']))
                    
            # AI tư duy bước tiếp theo
            phan_hoi = self.llm_kem_cong_cu.invoke(self.tin_nhan)
        
        # Bóc tách kết quả cuối cùng
        noi_dung_tra_loi = phan_hoi.content
        if isinstance(noi_dung_tra_loi, list):
            noi_dung_tra_loi = noi_dung_tra_loi[0]['text']
        
        self.tin_nhan.append(phan_hoi)
        return noi_dung_tra_loi

    def chay_giao_dien(self):
        """Vòng lặp Console UI tương tác với người dùng."""
        print("Trợ lý AI (Bản OOP) đã sẵn sàng! 🤖 (Gõ 'Exit' để kết thúc)")
        while True:
            cau_hoi = input("\nBạn: ")
            if cau_hoi.lower() == 'exit':
                print("Tạm biệt!")
                break
            
            cau_tra_loi = self.xu_ly_tin_nhan(cau_hoi)
            print("Trợ lý AI:", cau_tra_loi)


def main(environ: Mapping[str, str] = os.environ) -> int:
    # 1. Đọc cấu hình
    try:
        api_key = read_api_key(environ)
    except MissingApiKeyError:
        print(MISSING_API_KEY_HINT, file=sys.stderr)
        return EXIT_MISSING_API_KEY

    model_name = read_model_name(environ)
    
    # 2. Khởi tạo đối tượng Trợ Lý và chạy
    try:
        tro_ly = TroLyGhiChu(api_key=api_key, model_name=model_name)
        tro_ly.chay_giao_dien()
    except ChatGoogleGenerativeAIError as error:
        if is_quota_exhausted(error):
            print(describe_quota_error(model_name), file=sys.stderr)
            return EXIT_QUOTA_EXHAUSTED
        raise

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())