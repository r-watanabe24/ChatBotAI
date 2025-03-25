import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "cl-tohoku/bert-base-japanese"
SAVE_DIR = "./saved_model_pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6).to(device)

dataset = load_dataset("json", data_files="faq_data.json")

def preprocess(examples):
    return tokenizer(examples["question"], truncation=True, padding=True, max_length=128)

tokenized = dataset.map(preprocess, batched=True)

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["train"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
