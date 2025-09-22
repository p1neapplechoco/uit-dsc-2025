import json, re, torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import ast


class EE:
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

    def __call__(self, text):
        system = (
            "Bạn là một chuyên gia trích xuất thực thể bằng tiếng Việt. "
            "Hãy xác định và phân loại các thực thể trong đoạn văn dưới đây."
        )
        shots = [
            (
                "Hưng và Lan là đôi bạn thân. Hưng rất là đẹp trai.",
                "[{'text': 'Hưng', 'type': 'PERSON'}, "
                "{'text': 'Lan', 'type': 'PERSON'}]",
            ),
            (
                "Hà Nội là thủ đô của Việt Nam.",
                "[{'text': 'Hà Nội', 'type': 'LOCATION'}, "
                "{'text': 'Việt Nam', 'type': 'LOCATION'}]",
            ),
            (
                "Apple Inc. là một công ty công nghệ lớn.",
                "[{'text': 'Apple Inc.', 'type': 'ORGANIZATION'}]",
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
            {"role": "user", "content": text},
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
            max_new_tokens=1200,
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
