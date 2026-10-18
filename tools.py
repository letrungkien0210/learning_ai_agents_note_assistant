from langchain_core.tools import tool

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