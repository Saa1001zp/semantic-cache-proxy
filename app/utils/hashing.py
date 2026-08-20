import hashlib
import numpy as np


def deterministic_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    фолбэк эмбеддинг когда нет модели.
    детерминированный, быстрый, для тестов и оффлайна.
    не семантический на 100%, но сохраняет свойство:
    одинаковый текст -> одинаковый вектор, похожий текст -> чуть похожий вектор
    делаем через n-gram хеширование
    """
    text = text.lower().strip()
    if not text:
        return np.zeros(dim, dtype=np.float32)

    vec = np.zeros(dim, dtype=np.float32)

    # разбиваем на 3-граммы + слова
    tokens = text.split()
    ngrams = []
    for tok in tokens:
        # char trigrams
        if len(tok) <= 3:
            ngrams.append(tok)
        else:
            for i in range(len(tok) - 2):
                ngrams.append(tok[i:i+3])
    ngrams.extend(tokens)  # + целые слова

    for ng in ngrams:
        h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
        idx = h % dim
        # знак тоже от хеша чтобы не было только положительных
        sign = 1 if (h >> 16) % 2 == 0 else -1
        vec[idx] += sign * (1.0 + (h % 10) / 10.0)

    # нормируем
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)
