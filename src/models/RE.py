from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import ast, json, re, torch


class RE:
    def __init__(self, model_name, device="auto"):
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
                dtype=torch.bfloat16,
            ).eval()
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    dtype=torch.float16,
                )
                .to(self.device)
                .eval()
            )

    @staticmethod
    def __clean_json(text):
        try:
            return json.loads(text)

        except Exception:
            entities = ast.literal_eval(text)
            return entities

    def __call__(self, text, entities):
        system = (
            "Bạn là công cụ TRÍCH XUẤT QUAN HỆ tiếng Việt.\n"
            "Chỉ trả về MỘT MẢNG JSON (không markdown/không giải thích/không bọc JSON trong chuỗi).\n"
            'Mỗi phần tử: {"head","tail","type","value", "desc"}.\n'
            "Sử dụng CHÍNH XÁC các entity đã cho làm head/tail; không tạo entity mới.\n"
            "Thực sự sử dụng các quan hệ có ý nghĩa giữa các thực thể đang được xét.\n"
            "Nếu không có quan hệ cho một cặp, hãy bỏ qua cặp đó.\n"
            "type là nhãn ngắn (FRIENDSHIP, EMOTION, QUALITY, LOCATION, WORK_FOR, ...).\n"
            "value là cụm nguyên văn ngắn nằm trong văn bản (ví dụ 'chúc mừng', 'rất to', 'ở').\n"
            "desc là cụm nguyên văn bổ sung (nếu có) để làm rõ ý nghĩa của value.\n"
            "JSON hợp lệ, không kèm text khác."
        )

        shots = [
            (
                "Lan gặp Minh ở Hà Nội. Minh rất là đẹp trai.",
                "[{'text': 'Lan', 'type': 'PERSON'}, "
                "{'text': 'Minh', 'type': 'PERSON'}]",
                "[{'head': 'Lan', 'tail': 'Minh', 'type': 'ACTIVITY', 'value': 'gặp', 'desc': 'ở Hà Nội'},"
                " {'head': 'Minh', 'tail': 'Minh', 'type': 'QUALITY', 'value': 'rất là đẹp trai', 'desc': ''}],",
            ),
            (
                "Hà Nội là thủ đô của Việt Nam.",
                "[{'text': 'Hà Nội', 'type': 'LOCATION'}, "
                "{'text': 'Việt Nam', 'type': 'LOCATION'}]",
                "[{'head': 'Hà Nội', 'tail': 'Việt Nam', 'type': 'LOCATION', 'value': 'là thủ đô của', 'desc': ''}]",
            ),
        ]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": shots[0][0]},
            {"role": "assistant", "content": shots[0][2]},
            {"role": "user", "content": shots[1][0]},
            {"role": "assistant", "content": shots[1][2]},
            {
                "role": "user",
                "content": f"Đoạn văn:\n{text}\n---\nEntities:\n{entities}\n---\nHãy áp dụng đúng quy tắc trên.",
            },
        ]

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
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][len(input_ids[0]) :], skip_special_tokens=True
        )

        return self.__clean_json(response)
