import json, re, torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import ast


class EE:
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

    @staticmethod
    def extract_first_json_array(s: str):
        # Tóm đúng mảng JSON đầu tiên để tránh rác
        m = re.search(r"\[\s*(?:\{.*?\})\s*(?:,\s*\{.*?\}\s*)*\]", s, flags=re.S)
        if not m:
            raise ValueError("No JSON array found")
        return json.loads(m.group(0))

    @staticmethod
    def filter_entities_in_text(entities, src: str):
        src_norm = src
        out, seen = [], set()
        for e in entities:
            t = e.get("text", "")
            typ = e.get("type", "")
            if t and t in src_norm:
                key = (t, typ.upper())
                if key not in seen:
                    seen.add(key)
                    e["type"] = typ.upper()
                    out.append(e)
        return out

    def __call__(self, text):
        system = (
            "Bạn là chuyên gia NER tiếng Việt.\n"
            "NHIỆM VỤ:\n"
            "- Chỉ trích xuất thực thể XUẤT HIỆN TRONG ĐOẠN <doc>...</doc> phía dưới (dạng chuỗi con chính xác).\n"
            "- Không thêm thực thể từ ví dụ ở trên hay kiến thức ngoài văn bản.\n"
            '- Trả về DUY NHẤT một mảng JSON: [{"text": ..., "type": ...}]. Không giải thích.'
        )

        shots = [
            (
                "Hưng và Lan là đôi bạn thân.",
                r"""[{"text":"Hưng","type":"PERSON"},{"text":"Lan","type":"PERSON"}]""",
            ),
            (
                "Hà Nội là thủ đô của Việt Nam.",
                r"""[{"text":"Hà Nội","type":"LOCATION"},{"text":"Việt Nam","type":"LOCATION"}]""",
            ),
            (
                "Công ty ABC là một doanh nghiệp công nghệ.",
                r"""[{"text":"Công ty ABC","type":"ORGANIZATION"}]""",
            ),
        ]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": shots[0][0]},
            {"role": "assistant", "content": shots[0][1]},
            {"role": "user", "content": shots[1][0]},
            {"role": "assistant", "content": shots[1][1]},
            {"role": "user", "content": shots[2][0]},
            {"role": "assistant", "content": shots[2][1]},
            {"role": "user", "content": f"<doc>\n{text}\n</doc>"},
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

        ents = self.extract_first_json_array(response)
        ents = self.filter_entities_in_text(ents, text)
        return ents
