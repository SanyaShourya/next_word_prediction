from preprocessing import Preprocessing
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Predict:
    def __init__(self,obj:Preprocessing):
        self.tokenizer=obj.tokenizer
        self.input_sequences=obj.input_sequences
        self.max_len=max([len(seq) for seq in self.input_sequences])
        self.index_word = {v: k for k, v in self.tokenizer.word_index.items()}
        
    def predict_next_words(self,model, seed_text, next_words=15):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted = np.argmax(predicted_probs)

            output_word = self.index_word.get(predicted, '')
            if output_word == "<OOV>":
                break
            seed_text += " " + output_word
        return seed_text

