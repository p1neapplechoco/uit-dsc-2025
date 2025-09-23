import json, re, torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import ast


class EE:
    def __init__(self, model_name, device="auto", torch_dtype=torch.bfloat16):
        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Load model without forcing device_map so we can control placement
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        # Move to target device when not CPU (CPU is default)
        if self.device.type != "cpu":
            self.model = self.model.to(device=self.device)
        self.model.eval()

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
            "Bạn là chuyên gia thực thể trong tiếng Việt.\n"
            "NHIỆM VỤ:\n"
            "- Chỉ trích xuất TOÀN BỘ thực thể XUẤT HIỆN TRONG ĐOẠN <doc>...</doc> phía dưới (dạng chuỗi con chính xác).\n"
            "- Không thêm thực thể từ ví dụ ở trên hay kiến thức ngoài văn bản.\n"
            "- Hãy cố gắng trích xuất đầy đủ, KHÔNG BỎ SÓT thực thể nào.\n"
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

        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        # Move inputs to the correct device
        enc = enc.to(self.device)

        eos_ids = [self.tokenizer.eos_token_id]
        try:
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None:
                eos_ids.append(im_end_id)
        except Exception:
            pass

        with torch.no_grad():
            out = self.model.generate(
                enc,
                max_new_tokens=1200,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=eos_ids,
            )

        gen_ids = out[0, enc.shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if text.startswith("```"):
            text = text.strip("` \n")
            lines = [
                ln
                for ln in text.splitlines()
                if not ln.strip().startswith(("```", "text", "json"))
            ]
            text = "\n".join(lines).strip()

        ents = self.extract_first_json_array(text)
        ents = self.filter_entities_in_text(ents, text)

        return ents
