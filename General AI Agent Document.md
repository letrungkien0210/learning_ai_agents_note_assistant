# Cẩm Nang Xây Dựng AI Agent Từ Con Số Không 🤖

Tài liệu này tổng hợp các kiến thức nền tảng và mã nguồn thực tế để xây dựng một **AI Agent** có khả năng quản lý ghi chú bằng Python, LangChain và Gemini API.

> 🔗 **Mã nguồn tham khảo (Code Example Repository):** [learning_ai_agents_note_assistant](https://github.com/letrungkien0210/learning_ai_agents_note_assistant)

---

## 1. AI Agent là gì? 💡

Khác với các mô hình ngôn ngữ lớn (**LLM** - Large Language Model) truyền thống chỉ hoạt động theo cơ chế "nhận văn bản - trả về văn bản", một **AI Agent** là một hệ thống có khả năng tự chủ:
* 🧠 **Suy luận (Reasoning):** Hiểu được ý định của người dùng từ ngôn ngữ tự nhiên và tự lên kế hoạch giải quyết vấn đề.
* 🛠️ **Hành động (Acting):** Biết cách lựa chọn và sử dụng các công cụ bên ngoài (**Tools**) để tương tác với thế giới thực (ví dụ: đọc/ghi file, tìm kiếm web, gọi API).

---

## 2. Các Thành Phần Cốt Lõi (Core Components) 🧩

Một AI Agent hoàn chỉnh được cấu tạo từ 3 bộ phận chính:

| Thành phần | Vai trò | Triển khai trong code |
| :--- | :--- | :--- |
| **1. Bộ não (Brain / LLM)** 🧠 | Trung tâm xử lý ngôn ngữ, phân tích yêu cầu và ra quyết định chọn công cụ. | Mô hình `gemini-1.5-pro` kết hợp với hàm `.bind_tools()`. |
| **2. Công cụ (Tools)** 🛠️ | "Đôi tay" giúp Agent thực thi các tác vụ cụ thể ở môi trường bên ngoài. | Các hàm Python có gắn nhãn `@tool` (`them_ghi_chu`, `doc_ghi_chu`). |
| **3. Trí nhớ (Memory)** 💾 | Lưu trữ ngữ cảnh và lịch sử trò chuyện để duy trì hội thoại liên tục. | Danh sách `tin_nhan = []` chứa các `HumanMessage`, phản hồi của AI và `ToolMessage`. |

---

## 3. Cơ Chế Vòng Lặp ReAct (Reasoning + Acting) 🔄

**ReAct** là viết tắt của **Re**asoning (Suy luận) và **Act**ing (Hành động). Đây là quy trình giúp AI Agent phối hợp nhịp nhàng giữa việc suy nghĩ và làm việc qua 4 bước:

1. 💭 **Thought (Suy nghĩ):** Người dùng gửi một `HumanMessage` (ví dụ: *"Chiều nay đi siêu thị mua rau"*). LLM phân tích câu nói và nhận ra cần phải dùng công cụ `them_ghi_chu`.
2. ⚡ **Action (Hành động):** LLM không trả lời ngay mà phát ra một yêu cầu gọi công cụ (`tool_calls`) chứa tên hàm và tham số (`args`). Code Python sẽ tra cứu hàm trong **Dictionary** (`danh_ba_cong_cu`) và chạy lệnh `.invoke()`.
3. 👁️ **Observation (Quan sát):** Kết quả sau khi chạy hàm Python (ví dụ: *"Đã lưu ghi chú thành công!"*) được đóng gói vào một `ToolMessage` và gửi ngược lại cho LLM.
4. 💬 **Final Response (Phản hồi cuối cùng):** LLM đọc `ToolMessage`, biết việc đã hoàn tất và tạo ra câu trả lời tự nhiên để giao tiếp với người dùng.

---

## 4. Mã Nguồn Thực Hành Hoàn Chỉnh 💻

> 📦 **Repository mã nguồn đầy đủ:** <https://github.com/letrungkien0210/learning_ai_agents_note_assistant>

Dự án của chúng ta gồm 2 file chính hoạt động cùng nhau:

### File 1: `tools.py` (Định nghĩa các công cụ 🛠️)
Nơi chứa các hàm Python giúp can thiệp vào file `todo.txt`. Chú thích (docstring) bên trong hàm rất quan trọng vì LLM sẽ đọc dòng này để hiểu công dụng của từng hàm.

```python
from langchain_core.tools import tool

@tool
def them_ghi_chu(noi_dung: str) -> str:
    """Thêm một công việc hoặc ghi chú mới vào danh sách."""
    # Mở file ở chế độ 'a' (append - ghi tiếp vào cuối file)
    with open('todo.txt', 'a', encoding='utf-8') as file:
        file.write(noi_dung + "\n")
    return "Đã lưu ghi chú thành công!"

@tool
def doc_ghi_chu() -> str:
    """Đọc và trả về danh sách các công việc đã lưu."""
    # Mở file ở chế độ 'r' (read - đọc nội dung)
    with open('todo.txt', 'r', encoding='utf-8') as file:
        noi_dung = file.read()
    
    if not noi_dung:
        return "Danh sách công việc đang trống."
    
    return noi_dung
```

### File 2: `brain.py` (Bộ não, Trí nhớ và Vòng lặp ReAct 🧠)
Nơi kết nối LLM với Tools, duy trì bộ nhớ bằng vòng lặp `while True` và tự động hóa việc gọi hàm bằng **Dictionary**.

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from tools import them_ghi_chu, doc_ghi_chu

# 1. Cấu hình API Key
os.environ["GOOGLE_API_KEY"] = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"

# 2. Khởi tạo Bộ não (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# 3. Kết nối công cụ vào Bộ não
danh_sach_cong_cu = [them_ghi_chu, doc_ghi_chu]
llm_kem_cong_cu = llm.bind_tools(danh_sach_cong_cu)

# 4. Tạo Danh bạ công cụ (Dictionary) để tự động hóa
danh_ba_cong_cu = {
    "them_ghi_chu": them_ghi_chu,
    "doc_ghi_chu": doc_ghi_chu
}

# 5. Khởi tạo Trí nhớ (Memory) ngoài vòng lặp
tin_nhan = []
print("Trợ lý AI đã sẵn sàng! (Gõ 'thoát' để kết thúc)")

# 6. Vòng lặp hội thoại liên tục và xử lý ReAct
while True:
    cau_hoi = input("\nBạn: ")
    
    if cau_hoi.lower() == 'thoát':
        print("Tạm biệt!")
        break
        
    # Lưu câu hỏi của người dùng vào bộ nhớ
    tin_nhan.append(HumanMessage(content=cau_hoi))
    
    # AI suy luận (Reasoning)
    phan_hoi = llm_kem_cong_cu.invoke(tin_nhan)
    
    # Nếu AI quyết định gọi công cụ (Acting)
    for yeu_cau in phan_hoi.tool_calls:
        ten_cong_cu = yeu_cau['name']
        cong_cu_duoc_chon = danh_ba_cong_cu[ten_cong_cu]
        
        # Thực thi công cụ
        ket_qua = cong_cu_duoc_chon.invoke(yeu_cau['args'])
        print(f"Hệ thống báo: {ket_qua}")
        
        # Cập nhật kết quả vào bộ nhớ (Observation)
        tin_nhan.append(phan_hoi) 
        tin_nhan.append(ToolMessage(content=ket_qua, tool_call_id=yeu_cau['id']))
        
        # AI đọc kết quả và phản hồi tự nhiên cho người dùng
        phan_hoi_cuoi_cung = llm_kem_cong_cu.invoke(tin_nhan)
        print("Trợ lý AI:", phan_hoi_cuoi_cung.content)
```

---

## 5. Phân Biệt Single-Agent (Tool Calling) và Multi-Agent ⚖️

Khi xây dựng các hệ thống AI, ranh giới giữa việc dùng công cụ (Tool) và dùng tác tử (Agent) thường dễ gây nhầm lẫn do cách các thư viện lập trình đóng gói chúng. Dưới đây là cách phân biệt cốt lõi dựa trên "Bộ não" (LLM).

### 5.1. Single-Agent với Tool Calling (Cơ chế cơ bản)
* **Bản chất:** Hệ thống chỉ có **1 Bộ não duy nhất** (LLM Router) phải tự suy nghĩ và tự cầm các công cụ để làm việc.
* **Tool (Công cụ) là gì?** Là các đoạn mã lập trình tĩnh, vô tri (như lệnh `if/else` hoặc lệnh mở file). Tool không có khả năng tự suy luận, chỉ nhận đầu vào và nhả đầu ra một cách máy móc.
* **Ví dụ:** LLM nhận câu nói "Khuya nay mua thuốc hạ sốt ngay" -> Quyết định gọi hàm `kiem_duyet_cong_viec`. Hàm Python dùng lệnh `in` để tìm chữ "ngay", thấy khớp nên trả về nhãn `[Khẩn cấp] 🚨`. Nếu câu nói không có chữ "ngay", hàm sẽ máy móc đánh giá là bình thường.

### 5.2. Hệ thống Multi-Agent (Kiến trúc nâng cao)
* **Bản chất:** Hệ thống có **Nhiều Bộ não (LLMs)** phối hợp làm việc như một công ty.
* **Agent (Tác tử) là gì?** Một Agent thực thụ phải bao gồm: `LLM độc lập` + `System Prompt riêng` (Ví dụ: "Bạn là chuyên gia y tế đánh giá mức độ rủi ro...").
* **Ví dụ:** LLM Router (Sếp) nhận câu nói "Khuya nay đi mua thuốc hạ sốt". Sếp không gọi code tĩnh, mà giao việc cho Audit Agent (Nhân viên). LLM của nhân viên tự suy luận: *"Việc mua thuốc hạ sốt vào lúc nửa đêm là tình huống y tế nguy hiểm"* và tự động dán nhãn `[Khẩn cấp] 🚨` dù không hề có từ khóa "gấp" hay "ngay".

### 🌟 Tóm lại
* **Tool Calling:** LLM gọi **Code tĩnh** (Như người dùng máy tính bỏ túi).
* **Multi-Agent:** LLM gọi **LLM khác** (Như Giám đốc giao việc cho Trưởng phòng). 
> 💡 *Lưu ý: Trong LangChain, một "Agent Nhân viên" thường được bọc bên trong hình thức của một `@tool` để Agent Sếp dễ dàng nhận diện và gọi tên.*