import os
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainATE")

MODEL_NAME = "bert-base-uncased"
DATA_PATH = "data/processed/ate_train.json"
OUTPUT_DIR = "models/ate_model"

def train_ate_model():
    if not os.path.exists(DATA_PATH):
        logger.info(f"Data file '{DATA_PATH}' not found. Generating sample data...")
        from src.dataset_loader import create_sample_processed_data
        create_sample_processed_data()

    try:
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            DataCollatorForTokenClassification,
            TrainingArguments,
            Trainer
        )
        import evaluate
    except ImportError as e:
        logger.error(f"Missing dependency for training: {e}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)
    label_list = ["O", "B-Aspect", "I-Aspect"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_align_labels(example):
        tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
        word_ids = tokenized.word_ids()
        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                labels.append(label2id[example["labels"][word_idx]])
        tokenized["labels"] = labels
        return {k: v for k, v in tokenized.items() if k != "polarity"}

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting ATE model training...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"ATE Model successfully saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    train_ate_model()
