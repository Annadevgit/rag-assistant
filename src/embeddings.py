from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    """Common interface for all embedding backends."""

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Train the backend on a reference corpus, if necessary."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Transform a list of texts into an embedding matrix (n_texts, dim)."""


class TfidfBackend(EmbeddingBackend):
    """TF-IDF backend — offline, fast, lexical similarity."""

    def __init__(self, max_features: int = 5000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words=None,  # the corpus is in French, so no relevant EN stopword list
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "The TF-IDF backend must be trained with .fit() before .encode()"
            )
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray()


class SentenceTransformerBackend(EmbeddingBackend):
    """
    sentence-transformers backend — semantic embeddings; requires an internet
    connection on the first model load (download from HuggingFace Hub, ~80 MB
    for the default model).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        # Nothing to train: the model is already pre-trained
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


def get_backend(name: str = "tfidf") -> EmbeddingBackend:
    """Factory for instantiating an embedding backend by name."""
    if name == "tfidf":
        return TfidfBackend()
    if name == "sentence-transformers":
        return SentenceTransformerBackend()
    raise ValueError(
        f"Unknown embedding backend: {name}. "
        "Available choices: 'tfidf', 'sentence-transformers'."
    )