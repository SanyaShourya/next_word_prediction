import pandas as pd,numpy as np
import re
from constants import VOCAB_SIZE
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
class Preprocessing:
    def __init__(self,df:pd.DataFrame):
        self.df=df
        self.vocab_size = VOCAB_SIZE
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.input_sequences = []
        self.sequence_limit_per_quote = 10  # Limit n-grams per quote
        self.total_sequence_limit = 100000  # Limit total sequences to prevent crash

        
    def clean_text(self,text):
            return re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
    def start_preprocessing(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and self.df[col].str.len().mean() > 20:
                quote_col = col
                break
            
        quotes = self.df[quote_col].astype(str)

        cleaned_quotes = [self.clean_text(q) for q in quotes]
        
        for line in cleaned_quotes:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, min(len(token_list), self.sequence_limit_per_quote)):
                n_gram = token_list[:i+1]
                input_sequences.append(n_gram)

                # Stop if we hit total limit
                if len(input_sequences) >= self.total_sequence_limit:
                    break
            if len(input_sequences) >= self.total_sequence_limit:
                break

        # Pad sequences
        max_len = max([len(seq) for seq in input_sequences])
        input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

        # Split into X and y
        input_sequences = np.array(input_sequences)
        X, y = input_sequences[:, :-1], input_sequences[:, -1]
        y = to_categorical(y, num_classes=self.vocab_size)
        
        return X,y
    

