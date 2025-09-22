import json, re, torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class COREF:
    def __init__(self, model_name):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def __call__(self, text: str) -> str:
        system = (
            "Bạn là công cụ thay đại từ bằng tên riêng cho tiếng Việt. "
            "Giữ NGUYÊN mọi chữ/dấu, chỉ thay đại từ ngôi 3 (anh ấy/anh ta/cô ấy/cô ta/"
            "chị ấy/ông ấy/bà ấy/họ/nó/…) bằng tên người gần nhất đã nhắc trước đó. "
            "Nếu không chắc đại từ ám chỉ ai, GIỮ NGUYÊN. "
            "Chỉ trả về đoạn văn đã chỉnh, KHÔNG lặp lại đề bài, KHÔNG in 'user' hay 'assistant'."
        )

        shots = [
            (
                "Reynolds sinh ra tại Rome. Anh ấy rất đẹp trai.",
                "Reynolds sinh ra tại Rome. Reynolds rất đẹp trai.",
            ),
            (
                "Lan gặp Minh ở Hà Nội. Cô ấy chúc mừng anh ấy.",
                "Lan gặp Minh ở Hà Nội. Lan chúc mừng Minh.",
            ),
        ]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": shots[0][0]},
            {"role": "assistant", "content": shots[0][1]},
            {"role": "user", "content": shots[1][0]},
            {"role": "assistant", "content": shots[1][1]},
            {
                "role": "user",
                "content": f"Đoạn văn:\n{text}\n---\nHãy áp dụng đúng quy tắc trên.",
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

        return text
