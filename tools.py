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