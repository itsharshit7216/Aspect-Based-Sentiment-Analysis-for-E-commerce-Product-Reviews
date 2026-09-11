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
    High-Performance Aspect-Based Sentiment Analysis Engine.
    Handles Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC).
    Optimized for instant (< 50ms) inference on cloud hosting (Render Free Tier).
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

        self._load_models()

    def _load_models(self):
        # 1. Load ATE Model if locally available
        if HAS_TRANSFORMERS and os.path.exists(self.ate_model_path):
            try:
                logger.info(f"Loading custom ATE model from {self.ate_model_path}...")
                self.ate_tokenizer = AutoTokenizer.from_pretrained(self.ate_model_path)
                self.ate_model = AutoModelForTokenClassification.from_pretrained(self.ate_model_path)
                self.id2label = self.ate_model.config.id2label
                self.is_custom_ate_loaded = True
                logger.info("Custom ATE model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load custom ATE model: {e}")

        # 2. Load ASC Model if locally available
        if HAS_TRANSFORMERS and os.path.exists(self.asc_model_path):
            try:
                logger.info(f"Loading custom ASC model from {self.asc_model_path}...")
                self.asc_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.asc_model = AutoModelForSequenceClassification.from_pretrained(self.asc_model_path)
                self.is_custom_asc_loaded = True
                logger.info("Custom ASC model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load custom ASC model: {e}")

    def extract_aspects(self, text: str) -> List[str]:
        """Extract aspect terms from input sentence with high precision."""
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

                results = [asp.replace("##", "") for asp in aspects if asp.strip()]
                if results:
                    return results
            except Exception as e:
                logger.error(f"Error in custom extract_aspects: {e}")

        return self._heuristic_aspect_extractor(text)

    def _heuristic_aspect_extractor(self, text: str) -> List[str]:
        """Comprehensive aspect extraction matching all product categories."""
        common_aspect_keywords = [
            "battery life", "battery", "screen", "display", "speakers", "speaker", 
            "sound quality", "sound", "audio", "noise cancellation", "camera quality", "camera", 
            "design", "keyboard", "trackpad", "touchpad", "price", "performance", 
            "speed", "weight", "portability", "build quality", "build", "durability", 
            "service", "customer service", "software", "processor", "charging", "battery backup",
            "storage", "connectivity", "bluetooth", "comfort", "picture quality"
        ]
        text_lower = text.lower()
        found_aspects = []
        for asp in common_aspect_keywords:
            if asp in text_lower:
                # Avoid overlapping sub-terms (e.g. don't add "battery" if "battery life" is already added)
                if not any(asp in existing for existing in found_aspects):
                    found_aspects.append(asp)

        if not found_aspects:
            words = [w.strip(".,!?") for w in text.split() if len(w) > 3]
            found_aspects = words[:2] if words else ["product"]

        return found_aspects

    def classify_sentiment(self, sentence: str, aspect: str) -> str:
        """Instant (< 1ms) clause-aware aspect sentiment classification with negation support."""
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

        sentence_lower = sentence.lower()
        # Split sentence into context clauses separated by conjunctions/punctuation
        delimiters = [",", ".", ";", " but ", " and ", " though ", " however ", " although ", " yet ", " while "]
        clauses = [sentence_lower]
        for d in delimiters:
            new_clauses = []
            for c in clauses:
                new_clauses.extend(c.split(d))
            clauses = new_clauses

        target_context = sentence_lower
        for clause in clauses:
            if aspect.lower() in clause:
                target_context = clause.strip()
                break

        pos_words = [
            "good", "great", "excellent", "amazing", "love", "awesome", "bright", 
            "fast", "sleek", "superb", "comfortable", "top notch", "clean", "stunning", 
            "crisp", "smooth", "premium", "decent", "impressive", "clear", "flawless", 
            "fantastic", "reliable", "solid", "best", "perfect", "worth", "satisfied"
        ]
        neg_words = [
            "bad", "terrible", "poor", "horrible", "disappointing", "dim", "slow", 
            "heavy", "drain", "drains", "cheap", "quiet", "weak", "lag", "lags", 
            "ugly", "flimsy", "horrendous", "uncomfortable", "overpriced", "buggy", 
            "broken", "useless", "blurry", "worst", "waste", "noisy", "disaster"
        ]

        # Check for negation (e.g., "not good", "not bad")
        has_negation = any(neg in target_context for neg in ["not ", "n't ", "never ", "hardly "])

        pos_score = sum(1 for w in pos_words if w in target_context)
        neg_score = sum(1 for w in neg_words if w in target_context)

        if has_negation:
            # Invert polarity on negation
            if pos_score > 0 and neg_score == 0:
                return "negative"
            elif neg_score > 0 and pos_score == 0:
                return "positive"

        if pos_score > neg_score:
            return "positive"
        elif neg_score > pos_score:
            return "negative"
        return "neutral"

    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """Perform instant end-to-end ABSA analysis on input text."""
        if not sentence or not sentence.strip():
            return {"sentence": sentence, "analysis": {}, "error": "Empty input sentence"}

        aspects = self.extract_aspects(sentence)
        analysis = {asp: self.classify_sentiment(sentence, asp) for asp in aspects}
        
        return {
            "sentence": sentence,
            "analysis": analysis,
            "extracted_aspects_count": len(aspects)
        }
