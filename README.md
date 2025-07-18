🧠 Calculating Attention Scores
A simple yet powerful implementation of attention mechanism fundamentals. This repository walks you through how attention scores are computed — the backbone of modern NLP models like Transformers and BERT.

📌 Overview
This project demonstrates how attention scores are calculated using the Scaled Dot-Product Attention method — a key component in transformer-based architectures.

We focus on the three main components:

Query (Q)

Key (K)

Value (V)

The attention mechanism computes a weighted sum of values, where the weights are determined by the compatibility of the queries with the corresponding keys.

📖 What is Attention?
In simple terms, attention allows a model to focus on relevant parts of the input sequence when producing an output. The core idea is to calculate how much each word (token) should "attend" to every other word in the sequence.

Mathematically:

text
Copy
Edit
Attention(Q, K, V) = softmax(QKᵀ / √d_k) * V
Where:

Q: Query matrix

K: Key matrix

V: Value matrix

d_k: Dimension of keys

🧰 Features
Compute raw dot-product attention

Apply scaling by √d_k

Apply softmax to normalize scores

Multiply with Value matrix to get final output

Step-by-step explanation in code comments

🗂️ Project Structure
bash
Copy
Edit
📁 calculating-attention-scores
├── main.py                # Core logic to calculate attention
├── attention_utils.py     # Modular functions for query, key, value, and attention
├── sample_data.py         # Sample vectors and toy examples
├── README.md              # Project overview
└── requirements.txt       # Dependencies (if any)
