import keras
from load_google_word2vec import load_google_w2v_model
from gensim import utils
import numpy as np
from gensim.models import Word2Vec

sentiment = {0: "negative", 1: "positive"}
STOP_LIST = set('for a of the and to in'.split())
WORD_NUMERIC_VECTOR_SIZE = 300
NO_OF_WORDS_TAKEN_FROM_REVIEW = 45


def document_to_batch(document, model: Word2Vec, time_steps):
    """
    Converts the document to its numeric representation
    :param document:
    :param model:
    :param time_steps: maximum number of words that will be taken into account during vector computation
    :return:
    """
    words_vectors_batch = []
    counter = 0
    for word in document:
        if word in model:
            words_vectors_batch.append(model.wv[word])
            counter += 1
        if counter >= time_steps:
            break

    for _ in range(counter, time_steps):
        words_vectors_batch.append(np.zeros(WORD_NUMERIC_VECTOR_SIZE))

    return np.array(words_vectors_batch)


def check_review():
    print("Wait for the Google Word2Vec model to be loaded...")
    w2v_model = load_google_w2v_model()
    net_model = keras.models.load_model("resources/sentiment_analyzer_model_45.h5")
    print("Word2Vec model loaded")

    while True:
        line = input("Type in your review or write \"quit\" and press ENTER in order to finish: ")

        if line == 'quit':
            break

        tokens_line = list(utils.tokenize(line, deacc=True, lower=True))
        document_review = list(filter(lambda x: x not in STOP_LIST, tokens_line))
        line_numeric = np.array([document_to_batch(document_review, w2v_model, NO_OF_WORDS_TAKEN_FROM_REVIEW)])

        print('I think this review is: ' + evaluate(net_model, line_numeric))
    print('Good bye')


def evaluate(model, document_batch):
    return sentiment[round(model.predict(document_batch)[0][0], 0)]


check_review()
