# 🧠 Next Word Prediction using LSTM (TensorFlow)

This project is a word-level **Next Word Prediction Model** built using **TensorFlow and LSTM** on a dataset of quotes. The model learns the context of words in sentences and predicts the next word(s) based on a given input phrase.

---

## 📁 Dataset

- A collection of motivational and philosophical quotes.
- Text cleaned to remove non-alphabet characters and converted to lowercase.

---

## ⚙️ Features

- Word-level tokenization (custom tokenizer)
- Vocabulary size is limited for efficient memory usage
- LSTM-based model trained to predict next word
- Capable of generating **up to 15 words** after an input phrase
- Handled completely on **Google Colab GPU**

---

## 🧪 Training

The model was trained with the following setup:

- **Text Preprocessing**:
  - Converted to lowercase
  - Removed all non-alphabetic characters using regex
  - Tokenized to word-level sequences
  - Generated n-gram sequences to predict the next word

- **Model Architecture**:
  - Embedding Layer
  - LSTM Layer
  - Dense Layer with Softmax Activation

- **Parameters**:
  - Optimizer: `Adam`
  - Loss Function: `categorical_crossentropy`
  - Epochs: `30`
  - Batch Size: `128`
  - Vocabulary Size: Limited (to avoid crashing Colab sessions)

- **Framework**: TensorFlow (Keras API)

---

## 📊 Results

- Achieved **66% accuracy**
- Final training loss ~ **1.5**
- Predicts **next 15 words** with decent semantic continuity
- Unknown words are handled using the `<UNK>` token

---

## 🚀 Example Usage

After training, run the prediction code and provide an input phrase:

```python
Enter a starting phrase: life is
Predicted continuation: life is a journey that must be embraced with courage and hope

