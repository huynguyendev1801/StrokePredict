import re

# Biểu thức chính quy
regex_pattern = r'.*(hello|hi|greetings|how are you today).*'

# Lấy chuỗi đầu vào từ người dùng
input_string = input("Nhập chuỗi của bạn: ")

# Kiểm tra xem chuỗi có chứa từ khóa nào trong danh sách không
match = re.match(regex_pattern, input_string, re.IGNORECASE)

# Nếu có, hiển thị câu trả lời là "hello"
if match:
    print("Xin chào các bạn")
