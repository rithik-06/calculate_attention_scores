import numpy as np
from attention import scaled_dot_product_attention

def main():
    np.random.seed(42)
    batch_size = 2
    seq_len_q = 3
    seq_len_k = 3
    depth = 4
    depth_v = 4

    query = np.random.rand(batch_size, seq_len_q, depth)
    key = np.random.rand(batch_size, seq_len_k, depth)
    value = np.random.rand(batch_size, seq_len_k, depth_v)
    mask = np.ones((batch_size, seq_len_q, seq_len_k))  # No masking

    output, attn_weights = scaled_dot_product_attention(query, key, value, mask)
    print("Query:\n", query)
    print("Key:\n", key)
    print("Value:\n", value)
    print("Output:\n", output)
    print("Attention Weights:\n", attn_weights)

if __name__ == "__main__":
    main() 