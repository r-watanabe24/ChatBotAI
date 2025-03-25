import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./saved_model_pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

label_to_answer = {
    0: "名前の変更はプロフィール設定から行えます。",
    1: "生年月日の変更はプロフィール設定から行ってください。",
    2: "出勤はホーム画面から「出勤」ボタンを押してください。",
    3: "退勤はホーム画面から「退勤」ボタンを押してください。",
    4: "ダークモードはマイページの設定から有効にできます。",
    5: "その質問にはお答えできません。",
}

def predict(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        label = torch.argmax(outputs.logits, dim=1).item()
    return label_to_answer.get(label, "該当する回答が見つかりません。")

while True:
    q = input("\n質問を入力してください（終了するには 'exit'）: ")
    if q.lower() == "exit":
        print("終了します。")
        break
    print("回答:", predict(q))
