from keras import Sequential
from keras.layers import LSTM, Dense
from data_generator import DataGenerator
from keras import callbacks
import numpy as np
import time

TIME_STEPS = 15
WORD_NUMERIC_VECTOR_SIZE = 300
DROPOUT = 0.4
RECURRENT_DROPOUT = 0.4
EPOCH_PATIENCE = 3
EPOCHS_NUMBER = 40
BATCH_SIZE = 20


class LstmNet():
    def __init__(self):
        self.callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_PATIENCE)]
        self.model = self.build_model()
        self.data_generator = self.create_data_generator()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(200, input_shape=(TIME_STEPS, WORD_NUMERIC_VECTOR_SIZE), dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def create_data_generator(self):
        return DataGenerator('/media/kaspiotr/Multimedia HDD/Sentiment_Analyzer_project_review_matrices/negative_reviews',
                             '/media/kaspiotr/Multimedia HDD/Sentiment_Analyzer_project_review_matrices/positive_reviews')

    def train_network(self):
        np.random.seed(7)

        print('Train...')
        start = time.time()
        self.model.fit_generator(self.data_generator.get_train_generator(BATCH_SIZE),
                                 epochs=EPOCHS_NUMBER,
                                 validation_data=self.data_generator.get_test_generator(BATCH_SIZE),
                                 steps_per_epoch=self.data_generator.get_test_samples_count() // BATCH_SIZE,
                                 callbacks=self.callbacks,
                                 validation_steps=self.data_generator.get_test_samples_count() // BATCH_SIZE,
                                 workers=8,
                                 use_multiprocessing=True)

        score, acc = self.model.evaluate_generator(self.data_generator.get_test_generator(BATCH_SIZE),
                                                   steps=self.data_generator.get_test_samples_count() // BATCH_SIZE)

        self.model.save("resources/sentiment_analyzer_model.h5")

        print('Score: %f' % score)
        print('Test accuracy: %f%%' % (acc * 100))
        end = time.time()
        print("time elapsed: ", end - start, "seconds")
