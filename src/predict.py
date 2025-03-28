import ollama
import re
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

def predict(message, model_type="0"):
    if model_type == "0":
        # 分類モデルで回答
        inputs = tokenizer(message, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            label = torch.argmax(outputs.logits, dim=1).item()
        return label_to_answer.get(label, "該当する回答が見つかりません。")

    elif model_type == "1":
        # LLMで回答
        query = (
            "アプリの使い方の質問に対して回答してください。\n"
            "アプリの機能は、\n"
            "1. マイページのプロフィール変更画面で名前、誕生日が可能。\n"
            "2. マイページの設定画面からダークモードの切り替えが可能。\n"
            "3. 出勤/退勤はホーム画面の 出勤ボタン、退勤ボタンから可能。\n"
            "それ以外の機能はないので該当しない場合は、\n"
            "その質問にはお答えできません　と答えてください。\n"
        )

        format_template = (
            "お問い合わせありがとうございます。\n"
            "回答の内容"
        )

        formatted_prompt = f"{query} 質問文：{message}\n\n回答のフォーマット：\n{format_template}\n"

        try:
            response = ollama.chat(
                model="stfate/llama3-elyza-jp-8b",
                messages=[{"role": "user", "content": formatted_prompt}],
                options={"max_tokens": 8192}
            )
            if 'message' in response and 'content' in response['message']:
                extracted_text = response['message']['content']
                extracted_text = re.sub(r'(?m)^\s*\n', '', extracted_text)
                extracted_text = re.sub(r'(\d+)\s*%', r'\1%', extracted_text)
                extracted_text = re.sub(r'(\d+)\s*円', r'\1円', extracted_text)
                return extracted_text
            else:
                return "エラー: LLM の応答形式が不正です。"
        except Exception as e:
            return f"エラー: {str(e)}"

    else:
        return "不正な model_type です（0=分類モデル, 1=LLM）"
        



model_type = input("使用するモデルを選んでください（0=分類モデル, 1=LLM）: ").strip()
print(f"{'分類モデル' if model_type == '0' else 'LLM'} を使用します。")

while True:
    q = input("\n質問を入力してください（終了するには 'exit'）: ")
    if q.lower() == "exit":
        print("終了します。")
        break
    print("回答:", predict(q, model_type))
