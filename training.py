import tensorflow as tf
class Training:
    def __init__(self,model,X,y):
        self.model=model
        self.X=X
        self.y=y
    def start_training(self):
        with tf.device('/GPU:0'):
            self.model.fit(self.X, self.y, epochs=250, batch_size=128, verbose=1)
