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

## 🔄 Training

The model was trained on a cleaned dataset of quotes using an LSTM-based neural network. Key training steps included:

- **Data Preparation**:
  - Lowercased all text
  - Removed all non-alphabetic characters
  - Tokenized text into word-level sequences
  - Created input sequences where each sequence predicts the next word

- **Model Details**:
  - Embedding layer to map words to dense vectors
  - LSTM layer to capture context and sequential information
  - Dense layer with softmax to predict the next word from the vocabulary

- **Training Parameters**:
  - Optimizer: `Adam`
  - Loss Function: `Categorical Crossentropy`
  - Batch Size: `128`
  - Epochs: `30`
  - Vocabulary size: `Limited to avoid memory issues`
  - Environment: Trained using **Google Colab** with GPU enabled

---

## 📊 Results

- Achieved a **training accuracy of ~66%**
- Final loss: **~1.5**
- Model successfully predicts next 15 words after a given starting phrase
- Handles unknown words using `<UNK>` token
- Avoids overfitting by managing sequence length and vocabulary size

---

## 🧠 Example Usage

You can generate next words interactively by entering a prompt:

```bash
Enter a starting phrase: life is
Predicted continuation: life is a journey that must be embraced with courage and hope

