import numpy as np
from typing import Tuple

def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the scaled dot-product attention.

    Args:
        query: shape (..., seq_len_q, depth)
        key: shape (..., seq_len_k, depth)
        value: shape (..., seq_len_v, depth_v)
        mask: (optional) shape (..., seq_len_q, seq_len_k)

    Returns:
        output: Attention output (..., seq_len_q, depth_v)
        attention_weights: Attention weights (..., seq_len_q, seq_len_k)
    """
    matmul_qk = np.matmul(query, key.transpose(-1, -2))  # (..., seq_len_q, seq_len_k)
    dk = key.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        scaled_attention_logits = np.where(mask == 0, -1e9, scaled_attention_logits)

    attention_weights = softmax(scaled_attention_logits, axis=-1)
    output = np.matmul(attention_weights, value)
    return output, attention_weights

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True) 