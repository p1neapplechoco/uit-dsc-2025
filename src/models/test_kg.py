from COREF import COREF
from EE import EE
from RE import RE

# Giả sử model_name là Qwen, bạn có thể thay đổi
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"  # Thay bằng model thực tế nếu khác

def create_knowledge_graph(text):
    """
    Chạy pipeline: COREF -> EE -> RE để xử lý văn bản và trích xuất entities và relations
    """
    # Bước 1: Coreference Resolution
    coref = COREF(MODEL_NAME)
    resolved_text = coref(text)
    print("Resolved Text:", resolved_text)

    # Bước 2: Entity Extraction
    ee = EE(MODEL_NAME)
    entities = ee(resolved_text)
    print("Entities:", entities)

    # Bước 3: Relation Extraction
    re_model = RE(MODEL_NAME)
    relations = re_model(resolved_text, entities)
    print("Relations:", relations)

    return entities, relations

if __name__ == "__main__":
    # Ví dụ văn bản test
    sample_text = "Lan gặp Minh ở Hà Nội. Minh rất là đẹp trai. Hà Nội là thủ đô của Việt Nam."

    # Tạo KG
    kg = create_knowledge_graph(sample_text)