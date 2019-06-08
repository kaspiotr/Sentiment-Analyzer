from os import listdir
import random
import numpy as np

TEST_DATA_PERCENTAGE = 30


class DataGenerator:
    def __init__(self, neg_reviews_dir_path, pos_reviews_dir_path):
        self.neg_reviews_dir_path = neg_reviews_dir_path
        self.pos_reviews_dir_path = pos_reviews_dir_path

        self.neg_reviews_files = [(neg_reviews_dir_path + "/" + f, 0) for f in listdir(self.neg_reviews_dir_path)]
        self.pos_reviews_files = [(pos_reviews_dir_path + "/" + f, 1) for f in listdir(self.pos_reviews_dir_path)]

        self.neg_train_samples_count = int(round(len(self.neg_reviews_files) * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
        self.pos_train_samples_count = int(round(len(self.pos_reviews_files) * (100 - TEST_DATA_PERCENTAGE) / 100, 0))
        self.neg_test_samples_count = len(self.neg_reviews_files) - self.neg_train_samples_count
        self.pos_test_samples_count = len(self.pos_reviews_files) - self.pos_train_samples_count

        self.train_samples_count = self.neg_train_samples_count + self.pos_train_samples_count
        self.test_samples_count = self.neg_test_samples_count + self.pos_test_samples_count

        self.train_files = self.neg_reviews_files[:self.neg_train_samples_count]
        self.train_files += self.pos_reviews_files[:self.pos_train_samples_count]
        random.shuffle(self.train_files)

        self.test_files = self.neg_reviews_files[self.neg_train_samples_count:]
        self.test_files += self.pos_reviews_files[self.pos_train_samples_count:]
        random.shuffle(self.test_files)

    def get_train_samples_count(self):
        return self.train_samples_count

    def get_test_samples_count(self):
        return self.test_samples_count

    def get_train_generator(self, batch_size):
        idx = 0
        while True:
            x = []
            y = []
            counter = 0
            while counter < batch_size:
                x.append(np.load(self.train_files[idx][0]))
                y.append(np.array(self.train_files[idx][1]))  # positive (1) or negative (0)
                idx = (idx + 1) % self.train_samples_count
                counter += 1
            yield (np.array(x), np.array(y))

    def get_test_generator(self, batch_size):
        idx = 0
        while True:
            x = []
            y = []
            counter = 0
            while counter < batch_size:
                x.append(np.load(self.test_files[idx][0]))
                y.append(np.array(self.test_files[idx][1]))  # positive (1) or negative (0)
                idx = (idx + 1) % self.test_samples_count
                counter += 1
            yield (np.array(x), np.array(y))
