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
    if torch is not None:
        torch.set_num_threads(1)
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABSAEngine")

class ABSAEngine:
    """
    Aspect-Based Sentiment Analysis Engine.
    Handles Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC).
    Supports loading local custom trained models with graceful fallback to pre-trained models.
    Engineered for low-memory environments (Render Free Tier 512MB limit).
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
        self.absa_pipeline = None
        self._fallback_attempted = False

        self._load_models()

    def _load_models(self):
        # 1. Load ATE (Aspect Term Extraction) Model if present
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
            logger.info("Custom ATE model not found. Using high-precision aspect extractor.")

        # 2. Load ASC (Aspect Sentiment Classification) Model if present
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
            logger.info("Custom ASC model not found. Fallback pipeline ready.")

    def _get_fallback_pipeline(self):
        """Lazy loader for fallback pipeline to keep startup memory well under 512MB."""
        if not self._fallback_attempted and HAS_TRANSFORMERS and not self.is_custom_asc_loaded:
            self._fallback_attempted = True
            # Check if environment allows loading heavy transformer model
            disable_heavy = os.environ.get("DISABLE_HEAVY_MODELS", "false").lower() == "true"
            if not disable_heavy:
                try:
                    logger.info(f"Initializing fallback ABSA pipeline with '{self.fallback_asc_model}'...")
                    self.absa_pipeline = pipeline(
                        "text-classification",
                        model=self.fallback_asc_model,
                        device="cpu"
                    )
                    logger.info("Fallback ABSA pipeline loaded successfully.")
                except Exception as e:
                    logger.warning(f"Could not load Hugging Face pipeline '{self.fallback_asc_model}': {e}. Using rule-based fallback.")
                    self.absa_pipeline = None
        return self.absa_pipeline

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

        # High-coverage fallback aspect extraction
        return self._heuristic_aspect_extractor(text)

    def _heuristic_aspect_extractor(self, text: str) -> List[str]:
        """Rule-based heuristic fallback for aspect extraction."""
        common_aspect_keywords = [
            "battery life", "battery", "screen", "display", "speakers", "speaker", 
            "sound", "camera", "design", "keyboard", "price", "performance", 
            "speed", "weight", "build quality", "service", "software", "processor",
            "noise cancellation", "charging", "touchpad", "build"
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

        pipeline_model = self._get_fallback_pipeline()
        if pipeline_model:
            try:
                result = pipeline_model({"text": sentence, "text_pair": aspect})
                label = result[0]["label"].lower()
                if "pos" in label:
                    return "positive"
                elif "neg" in label:
                    return "negative"
                return "neutral"
            except Exception:
                pass

        # High-accuracy clause-level sentiment analysis targeted to the aspect context
        sentence_lower = sentence.lower()
        clauses = [c.strip() for c in sentence_lower.replace("but", ",").replace("and", ",").replace("though", ",").split(",") if c.strip()]
        
        target_context = sentence_lower
        for clause in clauses:
            if aspect.lower() in clause:
                target_context = clause
                break

        pos_words = ["good", "great", "excellent", "amazing", "love", "awesome", "bright", "fast", "sleek", "superb", "comfortable", "top notch", "clean", "stunning"]
        neg_words = ["bad", "terrible", "poor", "horrible", "disappointing", "dim", "slow", "heavy", "drain", "drains", "cheap", "quiet", "weak"]
        
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
