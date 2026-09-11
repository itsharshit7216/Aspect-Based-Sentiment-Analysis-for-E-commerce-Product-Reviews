import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainASC")

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "models/asc_model"

def train_asc_model():
    """Train Aspect Sentiment Classification (ASC) model."""
    try:
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer
        )
    except ImportError as e:
        logger.error(f"Missing dependency for training: {e}")
        return

    sample_data = [
        {"sentence": "The screen is bright", "aspect": "screen", "label": 3},
        {"sentence": "Battery life is terrible", "aspect": "battery life", "label": 1},
        {"sentence": "The keyboard is okay", "aspect": "keyboard", "label": 2}
    ]

    dataset = Dataset.from_list(sample_data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            examples["aspect"],
            truncation=True,
            padding="max_length",
            max_length=64
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    logger.info("Starting ASC model training...")
    trainer.train()

    save_path = os.path.join(OUTPUT_DIR, "checkpoint-885")
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"ASC Model saved successfully to '{save_path}'")

if __name__ == "__main__":
    train_asc_model()
