"""
Три движка поиска ответов для сравнительного анализа:
  1. TfidfEngine        — базовый TF-IDF + лемматизация
  2. TfidfSynonymEngine — TF-IDF + лемматизация + словарь синонимов
  3. EmbeddingEngine    — sentence-transformers (нейросетевые эмбеддинги)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp_utils import lemmatize, load_knowledge_base

CONFIDENCE_THRESHOLD = 0.15


# ── Базовый класс ────────────────────────────────────────────
class BaseEngine:
    """Общий интерфейс для всех движков."""

    name: str = "base"

    def __init__(self, qa_data: dict):
        self.entries = qa_data["questions"]
        self.raw_questions = [e["question"] for e in self.entries]
        self.answers = [e["answer"] for e in self.entries]

    def find_answer(self, user_text: str) -> dict:
        raise NotImplementedError

    def _make_result(self, best_idx: int, best_score: float) -> dict:
        if best_score < CONFIDENCE_THRESHOLD:
            return {
                "answer": "Не нашёл подходящего ответа.",
                "score": best_score,
                "matched_question": None,
                "matched_id": None,
                "category": None,
            }
        entry = self.entries[best_idx]
        return {
            "answer": self.answers[best_idx],
            "score": best_score,
            "matched_question": self.raw_questions[best_idx],
            "matched_id": entry["id"],
            "category": entry.get("category", "—"),
        }


# ── 1. Базовый TF-IDF ────────────────────────────────────────
class TfidfEngine(BaseEngine):
    """TF-IDF + лемматизация, без синонимов."""

    name = "TF-IDF"

    def __init__(self, qa_data: dict):
        super().__init__(qa_data)
        self.lemmatized = [lemmatize(q, use_synonyms=False) for q in self.raw_questions]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.lemmatized)

    def find_answer(self, user_text: str) -> dict:
        lem = lemmatize(user_text, use_synonyms=False)
        if not lem.strip():
            return self._make_result(0, 0.0)
        vec = self.vectorizer.transform([lem])
        sims = cosine_similarity(vec, self.matrix)[0]
        idx = int(np.argmax(sims))
        return self._make_result(idx, float(sims[idx]))


# ── 2. TF-IDF + синонимы ─────────────────────────────────────
class TfidfSynonymEngine(BaseEngine):
    """TF-IDF + лемматизация + расширение синонимов."""

    name = "TF-IDF + синонимы"

    def __init__(self, qa_data: dict):
        super().__init__(qa_data)
        self.lemmatized = [lemmatize(q, use_synonyms=True) for q in self.raw_questions]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.lemmatized)

    def find_answer(self, user_text: str) -> dict:
        lem = lemmatize(user_text, use_synonyms=True)
        if not lem.strip():
            return self._make_result(0, 0.0)
        vec = self.vectorizer.transform([lem])
        sims = cosine_similarity(vec, self.matrix)[0]
        idx = int(np.argmax(sims))
        return self._make_result(idx, float(sims[idx]))


# ── 3. Продвинутый TF-IDF (char n-grams + word bigrams + синонимы) ──
class AdvancedTfidfEngine(BaseEngine):
    """
    Продвинутый TF-IDF:
    - Char n-grams (3-5) для обработки опечаток и морфологии
    - Word unigrams + bigrams для контекста
    - Синонимы
    - Два векторизатора, объединённый скор
    """

    name = "TF-IDF Advanced (char+word n-grams)"

    def __init__(self, qa_data: dict):
        super().__init__(qa_data)
        self.lemmatized = [lemmatize(q, use_synonyms=True) for q in self.raw_questions]

        # Vectorizer 1: word-level unigrams + bigrams
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True
        )
        self.word_matrix = self.word_vec.fit_transform(self.lemmatized)

        # Vectorizer 2: char-level n-grams (3-5) — ловит опечатки и части слов
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self.char_matrix = self.char_vec.fit_transform(self.lemmatized)

    def find_answer(self, user_text: str) -> dict:
        lem = lemmatize(user_text, use_synonyms=True)
        if not lem.strip():
            return self._make_result(0, 0.0)

        # Word similarity
        wv = self.word_vec.transform([lem])
        w_sims = cosine_similarity(wv, self.word_matrix)[0]

        # Char similarity
        cv = self.char_vec.transform([lem])
        c_sims = cosine_similarity(cv, self.char_matrix)[0]

        # Weighted combination: word 60% + char 40%
        combined = 0.6 * w_sims + 0.4 * c_sims
        idx = int(np.argmax(combined))
        return self._make_result(idx, float(combined[idx]))
