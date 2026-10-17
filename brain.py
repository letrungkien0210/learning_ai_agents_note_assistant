from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tools import them_ghi_chu
import os

# Cài đặt khóa API của bạn (ưu tiên lấy từ biến môi trường nếu có)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# 1. Khởi tạo "bộ não" Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# 2. Kết nối công cụ với mô hình AI
danh_sach_cong_cu = [them_ghi_chu]
llm_kem_cong_cu = llm.bind_tools(danh_sach_cong_cu)

# 3. Tạo tin nhắn yêu cầu từ người dùng
tin_nhan = [HumanMessage(content="Nhắc tôi chiều nay đi siêu thị mua rau")]

# 4. Yêu cầu Agent xử lý tin nhắn
phan_hoi = llm_kem_cong_cu.invoke(tin_nhan)

# In quyết định gọi công cụ của Agent ra màn hình
print(phan_hoi.tool_calls)