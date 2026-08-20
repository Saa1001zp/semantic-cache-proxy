import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """косинусная близость, 1 = идентично, 0 = ортогонально"""
    # защита от нулевых векторов
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    батчевый косинус: query (d,) vs matrix (n, d) -> (n,)
    быстрее чем в цикле, для поиска по кэшу
    """
    if matrix.size == 0:
        return np.array([])
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return np.zeros(matrix.shape[0])

    mat_norms = np.linalg.norm(matrix, axis=1)
    # avoid div by zero
    mat_norms = np.where(mat_norms == 0, 1e-9, mat_norms)

    dots = matrix @ query
    return dots / (mat_norms * q_norm)
