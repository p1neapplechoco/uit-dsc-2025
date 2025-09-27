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

            Quy tắc:
            1. Giữ nguyên ý nghĩa của câu trả lời ngắn
            2. Tạo câu trả lời đầy đủ, tự nhiên bằng tiếng Việt  
            3. Sử dụng thông tin từ context và câu hỏi để tạo câu hoàn chỉnh
            4. Không thêm thông tin không có trong context
            5. Câu trả lời phải ngắn gọn nhưng đầy đủ
            6. Chỉ trả về câu trả lời đầy đủ, không giải thích thêm

            Ví dụ:
            Context: "Năm 1954, kỷ niệm lần thứ 300 Hiệp ước Pereyaslav được tổ chức khắp nơi"
            Câu hỏi: "Năm 1954 là kỷ niệm lần thứ mấy của Hiệp ước Pereyaslav?"
            Trả lời ngắn: "300"
            Trả lời đầy đủ: "Năm 1954 là kỷ niệm lần thứ 300 của Hiệp ước Pereyaslav."

            Context: "Napoleon Bonaparte sinh năm 1769 tại Corsica"
            Câu hỏi: "Napoleon sinh năm nào?"
            Trả lời ngắn: "1769"
            Trả lời đầy đủ: "Napoleon sinh năm 1769."

            Context: "Thành phố Hà Nội là thủ đô của Việt Nam"
            Câu hỏi: "Đâu là thủ đô của Việt Nam?"
            Trả lời ngắn: "Hà Nội"
            Trả lời đầy đủ: "Hà Nội là thủ đô của Việt Nam."
        """

    def __call__(self, context: str, question: str, short_answer: str) -> str:
        try:
            user_prompt = f"""
                Context: {context}
                Câu hỏi: {question}
                Trả lời ngắn: {short_answer}

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
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Clean up response
            response = response.strip()
            if response.startswith("Trả lời đầy đủ:"):
                response = response[len("Trả lời đầy đủ:") :].strip()

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
        torch.cuda.empty_cache()
