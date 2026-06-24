import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings


class Embedder:

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        self._model = SentenceTransformer(settings.embedding_model)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
