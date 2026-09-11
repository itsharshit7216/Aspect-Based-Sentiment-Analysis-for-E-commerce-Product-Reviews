import os
import logging
from typing import Dict, List, Any

try:
    import torch

    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        AutoModelForSequenceClassification,
        pipeline
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABSAEngine")

class ABSAEngine:
    """
    Aspect-Based Sentiment Analysis Engine.
    Handles Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC).
    Supports loading local custom trained models with graceful fallback to Hugging Face pre-trained models.
    """

    def __init__(
        self,
        ate_model_path: str = "models/ate_model",
        asc_model_path: str = "models/asc_model/checkpoint-885",
        fallback_asc_model: str = "yangheng/deberta-v3-base-absa-v1.1"
    ):
        self.ate_model_path = ate_model_path
        self.asc_model_path = asc_model_path
        self.fallback_asc_model = fallback_asc_model
        
        self.is_custom_ate_loaded = False
        self.is_custom_asc_loaded = False

        self._load_models()

    def _load_models(self):
        # 1. Load ATE (Aspect Term Extraction) Model
        if HAS_TRANSFORMERS and os.path.exists(self.ate_model_path):
            try:
                logger.info(f"Loading custom ATE model from {self.ate_model_path}...")
                self.ate_tokenizer = AutoTokenizer.from_pretrained(self.ate_model_path)
                self.ate_model = AutoModelForTokenClassification.from_pretrained(self.ate_model_path)
                self.id2label = self.ate_model.config.id2label
                self.is_custom_ate_loaded = True
                logger.info("Custom ATE model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load custom ATE model from {self.ate_model_path}: {e}")
        else:
            logger.info(f"Custom ATE model path '{self.ate_model_path}' not found or transformers not installed. Using fallback extractor.")

        # 2. Load ASC (Aspect Sentiment Classification) Model
        if HAS_TRANSFORMERS and os.path.exists(self.asc_model_path):
            try:
                logger.info(f"Loading custom ASC model from {self.asc_model_path}...")
                self.asc_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.asc_model = AutoModelForSequenceClassification.from_pretrained(self.asc_model_path)
                self.is_custom_asc_loaded = True
                logger.info("Custom ASC model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load custom ASC model from {self.asc_model_path}: {e}")
        else:
            logger.info(f"Custom ASC model path '{self.asc_model_path}' not found.")

        # 3. Setup Fallback Pipeline if custom models are not loaded
        if HAS_TRANSFORMERS and not self.is_custom_asc_loaded:
            try:
                logger.info(f"Initializing fallback HuggingFace ABSA pipeline with '{self.fallback_asc_model}'...")
                self.absa_pipeline = pipeline("text-classification", model=self.fallback_asc_model)
                logger.info("Fallback ABSA pipeline loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load Hugging Face pipeline '{self.fallback_asc_model}': {e}. Using rule-based fallback.")
                self.absa_pipeline = None
        else:
            self.absa_pipeline = None

    def extract_aspects(self, text: str) -> List[str]:
        """Extract aspect terms from input sentence."""
        if self.is_custom_ate_loaded:
            try:
                tokens = self.ate_tokenizer(text.split(), is_split_into_words=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.ate_model(**tokens)
                predictions = torch.argmax(outputs.logits, dim=2)

                word_ids = tokens.word_ids()
                aspects, current = [], []

                for idx, word_id in enumerate(word_ids):
                    if word_id is None:
                        continue
                    token = tokens.tokens()[idx]
                    label = self.id2label[predictions[0][idx].item()]

                    if label.startswith("B-Aspect"):
                        if current:
                            aspects.append(" ".join(current))
                            current = []
                        current.append(token)
                    elif label.startswith("I-Aspect"):
                        current.append(token)
                    else:
                        if current:
                            aspects.append(" ".join(current))
                            current = []

                if current:
                    aspects.append(" ".join(current))

                return [asp.replace("##", "") for asp in aspects if asp.strip()]
            except Exception as e:
                logger.error(f"Error in custom extract_aspects: {e}")

        # Fallback Aspect Extractor (heuristic / noun chunk extraction)
        return self._heuristic_aspect_extractor(text)

    def _heuristic_aspect_extractor(self, text: str) -> List[str]:
        """Simple rule-based heuristic fallback for aspect extraction."""
        common_aspect_keywords = [
            "battery life", "battery", "screen", "display", "speakers", "speaker", 
            "sound", "camera", "design", "keyboard", "price", "performance", 
            "speed", "weight", "build quality", "service", "software", "processor"
        ]
        text_lower = text.lower()
        found_aspects = []
        for asp in common_aspect_keywords:
            if asp in text_lower and not any(asp in existing for existing in found_aspects):
                found_aspects.append(asp)

        if not found_aspects:
            words = [w.strip(".,!?") for w in text.split() if len(w) > 3]
            found_aspects = words[:2] if words else ["product"]

        return found_aspects

    def classify_sentiment(self, sentence: str, aspect: str) -> str:
        """Classify sentiment for a specific aspect in a sentence."""
        if self.is_custom_asc_loaded:
            try:
                inputs = self.asc_tokenizer(
                    sentence,
                    aspect,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )
                with torch.no_grad():
                    outputs = self.asc_model(**inputs)
                predicted_class = outputs.logits.argmax(dim=-1).item()

                label_map = {0: "negative", 1: "negative", 2: "neutral", 3: "positive"}
                return label_map.get(predicted_class, "neutral")
            except Exception as e:
                logger.error(f"Error in custom classify_sentiment: {e}")

        if self.absa_pipeline:
            try:
                result = self.absa_pipeline({"text": sentence, "text_pair": aspect})
                label = result[0]["label"].lower()
                if "pos" in label:
                    return "positive"
                elif "neg" in label:
                    return "negative"
                return "neutral"
            except Exception:
                pass

        # Clause-level sentiment fallback targeted to the aspect context
        sentence_lower = sentence.lower()
        clauses = [c.strip() for c in sentence_lower.replace("but", ",").replace("and", ",").split(",") if c.strip()]
        
        target_context = sentence_lower
        for clause in clauses:
            if aspect.lower() in clause:
                target_context = clause
                break

        pos_words = ["good", "great", "excellent", "amazing", "love", "awesome", "bright", "fast", "sleek", "superb"]
        neg_words = ["bad", "terrible", "poor", "horrible", "disappointing", "dim", "slow", "heavy", "drain", "drains"]
        
        pos_score = sum(1 for w in pos_words if w in target_context)
        neg_score = sum(1 for w in neg_words if w in target_context)

        if pos_score > neg_score:
            return "positive"
        elif neg_score > pos_score:
            return "negative"
        return "neutral"

    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """Perform end-to-end ABSA analysis on input text."""
        if not sentence or not sentence.strip():
            return {"sentence": sentence, "analysis": {}, "error": "Empty input sentence"}

        aspects = self.extract_aspects(sentence)
        analysis = {asp: self.classify_sentiment(sentence, asp) for asp in aspects}
        
        return {
            "sentence": sentence,
            "analysis": analysis,
            "extracted_aspects_count": len(aspects)
        }
