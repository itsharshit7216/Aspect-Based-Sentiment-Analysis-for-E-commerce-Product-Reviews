import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetLoader")

def download_semeval_dataset():
    """Download SemEval 2014 Task 4 ABSA dataset using kagglehub."""
    try:
        import kagglehub
        logger.info("Downloading dataset via kagglehub...")
        path = kagglehub.dataset_download("charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis")
        logger.info(f"Dataset downloaded successfully at: {path}")
        return path
    except Exception as e:
        logger.warning(f"Could not download dataset automatically: {e}")
        return None

def create_sample_processed_data(output_dir: str = "data/processed"):
    """Create sample processed data files for quick training/testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_ate_data = [
        {
            "tokens": ["The", "screen", "is", "bright", "and", "clear", "."],
            "labels": ["O", "B-Aspect", "O", "O", "O", "O", "O"],
            "polarity": ["positive"]
        },
        {
            "tokens": ["Battery", "life", "is", "terrible", "."],
            "labels": ["B-Aspect", "I-Aspect", "O", "O", "O"],
            "polarity": ["negative"]
        },
        {
            "tokens": ["The", "speakers", "sound", "amazing", "."],
            "labels": ["O", "B-Aspect", "O", "O", "O"],
            "polarity": ["positive"]
        }
    ]

    ate_path = os.path.join(output_dir, "ate_train.json")
    with open(ate_path, "w", encoding="utf-8") as f:
        json.dump(sample_ate_data, f, indent=2)

    logger.info(f"Sample dataset saved to {ate_path}")
    return ate_path

if __name__ == "__main__":
    path = download_semeval_dataset()
    if not path:
        create_sample_processed_data()
