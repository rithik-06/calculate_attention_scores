import numpy as np
import unittest
from attention import scaled_dot_product_attention

class TestScaledDotProductAttention(unittest.TestCase):
    def test_attention_shapes(self):
        batch_size = 1
        seq_len_q = 2
        seq_len_k = 2
        depth = 3
        depth_v = 3
        query = np.random.rand(batch_size, seq_len_q, depth)
        key = np.random.rand(batch_size, seq_len_k, depth)
        value = np.random.rand(batch_size, seq_len_k, depth_v)
        mask = np.ones((batch_size, seq_len_q, seq_len_k))
        output, attn_weights = scaled_dot_product_attention(query, key, value, mask)
        self.assertEqual(output.shape, (batch_size, seq_len_q, depth_v))
        self.assertEqual(attn_weights.shape, (batch_size, seq_len_q, seq_len_k))
        # Attention weights should sum to 1 along last axis
        np.testing.assert_allclose(np.sum(attn_weights, axis=-1), np.ones((batch_size, seq_len_q)), rtol=1e-5)

if __name__ == "__main__":
    unittest.main() 