from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from constants import VOCAB_SIZE
from preprocessing import Preprocessing
class Model:
    def __init__(self,X):
        self.X=X
    def make_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_shape=(self.X.shape[1],)))
        model.add(LSTM(100))
        model.add(Dense(VOCAB_SIZE, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
