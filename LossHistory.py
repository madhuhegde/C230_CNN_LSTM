from keras.callbacks import Callback 
class LossHistory(Callback):
    def on_train_batch_begin(self, batch, logs={}):
        self.losses = []

    def on_train_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
