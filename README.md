# 🧠 Next Word Prediction using LSTM (TensorFlow)

This project implements a deep learning model to predict the **next word(s)** in a given sentence using **LSTM (Long Short-Term Memory)** networks. The model is trained on a dataset of quotes and can predict up to **15 words** ahead.

---

## 🚀 Project Highlights

- Cleaned and preprocessed a large text dataset of quotes
- Tokenized text and created input sequences for training
- Built and trained an LSTM-based neural network using TensorFlow/Keras
- Handled large vocabularies by limiting vocabulary size
- Implemented word prediction loop to generate multiple words
- Achieved ~66% training accuracy
- Trained on Google Colab with GPU support without crashing

---

## 📂 Dataset

- **Type**: Collection of famous quotes
- **Format**: Text file with one quote per line
- **Preprocessing**:
  - Lowercased all text
  - Removed non-alphabet characters
  - Tokenized sentences into words
  - Limited vocabulary size (to avoid memory issues)

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab (GPU runtime)

---

## 📊 Model Architecture

- **Embedding Layer**: Converts word indices to dense vectors  
- **LSTM Layer**: Learns temporal dependencies in sequences  
- **Dense Layer (Softmax)**: Outputs probability distribution over vocabulary  

```python
Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    LSTM(units),
    Dense(vocab_size, activation='softmax')
])
