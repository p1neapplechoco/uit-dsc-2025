from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class COREF:
    def __init__(self, model_name, device="auto"):
        """
        Khởi tạo CoreferenceResolver sử dụng mô hình Qwen từ Hugging Face.

        Args:
            model_name: Name of Qwen model
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if torch.cuda.is_available() and device != "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).eval()
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
                .to(self.device)
                .eval()
            )

    def __call__(self, text):
        prompt = f"""
            Bạn là hệ thống chuẩn hoá tham chiếu (coreference resolution) cho tiếng Việt.

            NHIỆM VỤ
            - Viết lại đoạn văn dưới đây, thay TẤT CẢ đại từ/tham chiếu (anh ấy, cô ấy, ông, bà, nó, họ, người này, kẻ đó, v.v.)
            bằng đúng tên thực thể của chúng.
            - Khi thực thể là người, LUÔN dùng đầy đủ họ + tên (nếu trong văn bản có đủ thông tin để xác định).
            - TUYỆT ĐỐI KHÔNG rút gọn còn mỗi tên riêng (ví dụ chỉ “Albert” hoặc chỉ “Lan”).
            - KHÔNG thêm kiến thức ngoài văn bản (không bịa họ/tên nếu văn bản không nêu).
            - Giữ nguyên trật tự câu, dấu câu, viết hoa, ngoặc, ký hiệu, số, thuật ngữ.
            - Với quan hệ thân tộc kèm tên (Bố/Mẹ/Ông/Bà/Anh/Chị/Em + Tên), dùng dạng “<Chức danh> của <HỌ VÀ TÊN>”.
            - Nếu một tham chiếu mơ hồ (không thể gán chắc ≥ 0.9), GIỮ NGUYÊN tham chiếu đó, không đoán.

            VÍ DỤ
            - Gốc: "Lan gặp Nam. Cô ấy chào anh ấy."
            Kết quả đúng: "Lan gặp Nam. Lan chào Nam."
            (Không được viết "Lan chào anh Nam" hay "Lan chào anh ấy".)

            - Gốc: "Ông An đi công tác. Ông ấy đã đến Hà Nội."
            Kết quả đúng: "Ông An đi công tác. Ông An đã đến Hà Nội."

            - Gốc: "Bố Nam muốn anh thi điện."
            Kết quả đúng: "Bố của Nam muốn Nam thi điện."

            ĐOẠN GỐC:
            {text}

            CHỈ IN RA ĐOẠN VĂN ĐÃ THAY THẾ, KHÔNG GIẢI THÍCH:
            """

        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        try:
            if hasattr(self.model, "hf_device_map"):
                first_device = next(iter(self.model.hf_device_map.values()))
                input_ids = input_ids.to(first_device)
            else:
                input_ids = input_ids.to(self.device)
        except:
            input_ids = input_ids.to(self.device)

        attention_mask = (
            (input_ids != self.tokenizer.pad_token_id).long()
            if self.tokenizer.pad_token_id is not None
            else None
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                    if self.tokenizer.eos_token_id
                    else self.tokenizer.pad_token_id
                ),
            )

        response = self.tokenizer.decode(
            outputs[0][len(input_ids[0]) :], skip_special_tokens=True
        )

        return response
