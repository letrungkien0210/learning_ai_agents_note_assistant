from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

@tool
def them_ghi_chu(noi_dung: str) -> str:
    """
    Lưu một ghi chú hoặc công việc mới vào danh sách.
    Args:
        noi_dung (str): Nội dung của ghi chú cần lưu.
    """
    # Mở file 'todo.txt' ở chế độ 'a' (thêm tiếp) và hỗ trợ tiếng Việt (utf-8)
    with open('todo.txt', 'a', encoding='utf-8') as file:
        # Ghi nội dung vào file và thêm ký tự '\n' để xuống dòng
        file.write(noi_dung + "\n")
    
    # Trả về thông báo để Agent biết việc đã hoàn tất
    return "Đã lưu ghi chú thành công!"

@tool
def doc_ghi_chu() -> str:
    """Đọc và trả về danh sách các công việc đã lưu."""
    # Mở file ở chế độ 'r' (read - đọc)
    with open('todo.txt', 'r', encoding='utf-8') as file:
        # Lấy toàn bộ nội dung trong file ra
        noi_dung = file.read()
    
    # Nếu file trống, báo cho AI biết
    if not noi_dung:
        return "Danh sách công việc đang trống."
    
    return noi_dung

@tool
def kiem_duyet_cong_viec(noi_dung: str) -> str:
    """Kiểm duyệt và phân loại mức độ ưu tiên của công việc bằng AI."""
    # 1. Khởi tạo một LLM "Nhân viên" độc lập
    # Lấy key từ biến môi trường (như cách PowerShell đã cài đặt)
    api_key = os.environ.get("GOOGLE_API_KEY")
    llm_nhan_vien = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key)
    
    # 2. Cấp cho nhân viên này một "Vai trò" (System Prompt)
    huong_dan = """Bạn là một Chuyên gia Đánh giá Rủi ro và Quản lý Thời gian.
    Nhiệm vụ của bạn là đọc nội dung công việc và quyết định mức độ ưu tiên.
    
    Quy tắc phân loại:
    - Trả về CHÍNH XÁC chuỗi "[Khẩn cấp] 🚨" nếu công việc liên quan đến: sức khỏe, y tế (ví dụ: đau ốm, mua thuốc, tai nạn), thời hạn quá gấp, hoặc các tình huống bất thường vào ban đêm.
    - Trả về CHÍNH XÁC chuỗi "[Bình thường] 🟢" cho các công việc sinh hoạt, mua sắm, làm việc thông thường.
    
    Tuyệt đối chỉ trả về đúng 1 trong 2 nhãn trên, không giải thích gì thêm."""

    # 3. Giao tiếp với nhân viên
    tin_nhan_cho_nhan_vien = [
        SystemMessage(content=huong_dan),
        HumanMessage(content=f"Nội dung công việc cần đánh giá: {noi_dung}")
    ]

    # 4. Nhân viên tự suy nghĩ và trả kết quả
    phan_hoi = llm_nhan_vien.invoke(tin_nhan_cho_nhan_vien)

    # Trả kết quả của nhân viên về cho Sếp (Router)
    noi_dung_tra_loi = phan_hoi.content
    if isinstance(noi_dung_tra_loi, list):
        noi_dung_tra_loi = noi_dung_tra_loi[0]['text']

    return noi_dung_tra_loi.strip()

