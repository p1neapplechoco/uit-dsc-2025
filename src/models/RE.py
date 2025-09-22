from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import ast, json, re, torch


class RE:
    def __init__(self, model_name):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        enc = enc.to(self.model.device)
        eos_ids = [self.tok.eos_token_id]
        try:
            im_end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None:
                eos_ids.append(im_end_id)
        except Exception:
            pass
        out = self.model.generate(
            enc,
            max_new_tokens=2000,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=eos_ids,
        )
        gen_ids = out[0, enc.shape[1] :]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        if text.startswith("```"):
            text = text.strip("` \n")
            lines = [
                ln
                for ln in text.splitlines()
                if not ln.strip().startswith(("```", "text", "json"))
            ]
            text = "\n".join(lines).strip()

        return self.__clean_json(text)
