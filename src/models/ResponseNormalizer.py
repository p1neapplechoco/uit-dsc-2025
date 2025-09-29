import pandas as pd
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc


class ResponseNormalizer:
    def __init__(
        self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507", device: str = "auto"
    ):
        self.device = device
        print(f"Using device: {self.device}")

        try:
            print(f"Loading model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")

        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            raise

        self.system_prompt = """
            Bạn là một chuyên gia xử lý ngôn ngữ tiếng Việt. Nhiệm vụ của bạn là chuyển đổi các câu trả lời ngắn thành câu trả lời đầy đủ và tự nhiên dựa trên context và câu hỏi.

            QUY TẮC QUAN TRỌNG - PHẢI TUÂN THỦ NGHIÊM NGẶT:
            1. **TUYỆT ĐỐI KHÔNG ĐƯỢC thay đổi ý nghĩa của câu trả lời ngắn**
            2. **TUYỆT ĐỐI KHÔNG ĐƯỢC thay đổi, sửa chữa, hoặc điều chỉnh bất kỳ con số, số liệu, năm tháng, phần trăm, thống kê nào**
            3. **TUYỆT ĐỐI KHÔNG ĐƯỢC thêm, bớt, hoặc sửa đổi thông tin sự kiện**
            4. **CHỈ ĐƯỢC thêm cấu trúc ngữ pháp để tạo câu hoàn chỉnh, KHÔNG thay đổi nội dung**
            5. Giữ nguyên 100% thông tin từ câu trả lời ngắn
            6. Chỉ sử dụng thông tin có sẵn trong context và câu hỏi để tạo cấu trúc câu
            7. Không thêm bất kỳ thông tin mới nào không có trong câu trả lời ngắn
            8. Câu trả lời phải ngắn gọn nhưng đầy đủ về mặt ngữ pháp
            9. Chỉ trả về câu trả lời đầy đủ, không giải thích thêm
            10. Nếu câu trả lời ngắn đã đầy đủ, CHỈ điều chỉnh nhẹ cấu trúc ngữ pháp để tự nhiên hơn

            LƯU Ý ĐẶC BIỆT:
            - Nếu câu trả lời ngắn là "300" thì câu đầy đủ phải chứa chính xác số "300", không được thành "ba trăm" hay số khác
            - Nếu câu trả lời ngắn là "1969-1975" thì câu đầy đủ phải chứa chính xác "1969-1975"
            - Nếu câu trả lời ngắn là "25%" thì câu đầy đủ phải chứa chính xác "25%"
            - Vai trò của bạn CHỈ LÀ tạo cấu trúc câu, KHÔNG PHẢI chỉnh sửa nội dung
            
            Ví dụ:

            # Trường hợp câu trả lời là số/năm
            Context: "Năm 1954, kỷ niệm lần thứ 300 Hiệp ước Pereyaslav được tổ chức khắp nơi"
            Câu hỏi: "Năm 1954 là kỷ niệm lần thứ mấy của Hiệp ước Pereyaslav?"
            Trả lời ngắn: "300"
            Trả lời đầy đủ: "Năm 1954 là kỷ niệm lần thứ 300 của Hiệp ước Pereyaslav."

            # Trường hợp câu trả lời là tên riêng/địa danh
            Context: "Napoleon Bonaparte sinh năm 1769 tại đảo Corsica, thuộc Pháp"
            Câu hỏi: "Napoleon sinh ở đâu?"
            Trả lời ngắn: "Corsica"
            Trả lời đầy đủ: "Napoleon sinh ở đảo Corsica."

            # Trường hợp câu trả lời đã khá dài
            Context: ""Dân số Đông Đức giảm đều đặn trong suốt thời kỳ tồn tại của nó, từ 19 triệu người năm 1948 xuống còn 16 triệu năm 1990. Khoảng 4 triệu người trong dân số năm 1948 là những người Đức bị trục xuất từ các khu vực phía đông giới tuyến Oder-Neisse. Chủ yếu đây là hậu quả của sự di cư – khoảng một phần tư người Đông Đức đã rời bò đất nước trước khi Bức tường Berlin được hoàn thành năm 1961, và sau thời điểm đó, Đông Đức có tỷ lệ sinh rất thấp. Điều này trái ngược với Ba Lan, có dân số trong giai đoạn đó tăng từ 24 triệu năm 1950 (hơi lớn hơn Đông Đức) lên 38 triệu (gấp đôi dân số Đông Đức)."
            Câu hỏi: "Mặc dù dân số Đông Đức giảm đều đặn sau khi Bức tường Berlin được xây dựng vào năm 1961, tình hình dân số của nước này trong suốt thời kỳ tồn tại của nó là như thế nào?"
            Trả lời ngắn: "Dân số Đông Đức giảm đều đặn từ 19 triệu năm 1948 xuống còn 16 triệu năm 1990, chủ yếu do di cư và tỷ lệ sinh thấp sau khi Bức tường Berlin được xây dựng."
            Trả lời đầy đủ: "Dân số Đông Đức giảm đều đặn trong suốt thời kỳ tồn tại của nó, từ 19 triệu người năm 1948 xuống còn 16 triệu người năm 1990, chủ yếu do di cư – khoảng một phần tư người Đông Đức đã rời đất nước trước khi Bức tường Berlin được hoàn thành năm 1961, và sau thời điểm đó, Đông Đức có tỷ lệ sinh rất thấp."

            # Trường hợp câu trả lời là Yes/No
            Context: "Thủ đô của Việt Nam là Hà Nội. Hà Nội là thành phố lớn thứ hai của Việt Nam sau Thành phố Hồ Chí Minh."
            Câu hỏi: "Hà Nội có phải là thủ đô của Việt Nam không?"
            Trả lời ngắn: "Có"
            Trả lời đầy đủ: "Có, Hà Nội là thủ đô của Việt Nam."

            # Trường hợp câu trả lời là danh sách ngắn
            Context: "Các thành phố lớn của Việt Nam bao gồm Thành phố Hồ Chí Minh, Hà Nội, Đà Nẵng, và Hải Phòng."
            Câu hỏi: "Những thành phố lớn nào của Việt Nam?"
            Trả lời ngắn: "Thành phố Hồ Chí Minh, Hà Nội, Đà Nẵng, Hải Phòng"
            Trả lời đầy đủ: "Các thành phố lớn của Việt Nam bao gồm Thành phố Hồ Chí Minh, Hà Nội, Đà Nẵng, và Hải Phòng."

            # Trường hợp câu trả lời về nguyên nhân/lý do
            Context: "Chiến tranh thế giới thứ hai kết thúc vào năm 1945 do Đức và Nhật Bản đầu hàng sau khi Mỹ thả bom nguyên tử."
            Câu hỏi: "Tại sao Chiến tranh thế giới thứ hai kết thúc?"
            Trả lời ngắn: "Do Đức và Nhật Bản đầu hàng"
            Trả lời đầy đủ: "Chiến tranh thế giới thứ hai kết thúc do Đức và Nhật Bản đầu hàng sau khi Mỹ thả bom nguyên tử."
        """

    def __call__(self, context: str, question: str, short_answer: str) -> str:
        try:
            user_prompt = f"""
                Context: {context}
                Câu hỏi: {question}
                Trả lời ngắn: {short_answer}

                LƯU Ý: TUYỆT ĐỐI KHÔNG thay đổi bất kỳ số liệu, con số, phần trăm, năm tháng, hoặc thông tin sự kiện nào từ câu trả lời ngắn. CHỈ thêm cấu trúc ngữ pháp để hoàn thiện câu.

                Hãy viết lại thành câu trả lời đầy đủ:
            """

            # Create conversation format
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.05,  # Giảm thấp hơn để ổn định và ít thay đổi
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.8,  # Giảm để tập trung vào từ có xác suất cao
                    num_beams=1,  # Thêm để đảm bảo deterministic hơn
                )

            # Decode response
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Clean up response - cải thiện việc làm sạch
            response = response.strip()

            # Loại bỏ các prefix có thể xuất hiện
            prefixes_to_remove = [
                "Trả lời đầy đủ:",
                "Câu trả lời đầy đủ:",
                "Đáp án đầy đủ:",
                "Câu trả lời:",
            ]

            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix) :].strip()
                    break

            return response if response else short_answer

        except Exception as e:
            print(f"Error normalizing response: {str(e)}")
            return short_answer

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
