import logging
import csv
import numpy as np
from gensim import models
from gensim.utils import tokenize

NO_OF_WORDS_TAKEN_FROM_REVIEW = 45
_google_model = None
GOOGLE_NEWS_WORD_LIMIT = None
recommendations = []
reviews = []


def read_csv():
    with open('resources/steam_reviews.csv') as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        headers = []
        headers_read = False
        for row in read_csv:
            # print(row)
            if not headers_read:
                headers = row
                headers_read = True
                continue
            if row[headers.index('recommendation')] == 'Recommended':
                recommendations.append(1)
            else:
                recommendations.append(0)
            reviews.append(row[headers.index('review')])


def load_google_w2v_model(words_limit=GOOGLE_NEWS_WORD_LIMIT):
    global _google_model
    if _google_model is not None:
        return _google_model

    logging.info("Lading Google Word2Vec ...")
    google_model = models.KeyedVectors.load_word2vec_format(
        "resources/GoogleNews-vectors-negative300.bin",
        binary=True,
        limit=words_limit
    )
    logging.info("Finished Google Word2Vec Loading")
    _google_model = google_model
    return google_model


def write_vectors_matrix_to_file(idx, vectors_matrix):
    if recommendations[idx] == 0:
        np.save('/media/kaspiotr/Multimedia HDD/Sentiment_Analyzer_project_review_matrices/negative_reviews_%d/review%d' % (NO_OF_WORDS_TAKEN_FROM_REVIEW, idx), vectors_matrix)
    else:
        np.save('/media/kaspiotr/Multimedia HDD/Sentiment_Analyzer_project_review_matrices/positive_reviews_%d/review%d' % (NO_OF_WORDS_TAKEN_FROM_REVIEW, idx), vectors_matrix)


def create_review_matrix(reviews_list):
    idx = 0
    for review in reviews_list:
        tokens = tokenize(review, lowercase=True, deacc=False, encoding='utf8', errors='strict', to_lower=True,
                          lower=True)
        filtered_tokens = filter(lambda x: x in _google_model.vocab, tokens)
        filtered_tokens_list = list(filtered_tokens)
        filtered_tokens_len = len(filtered_tokens_list)
        if filtered_tokens_len >= NO_OF_WORDS_TAKEN_FROM_REVIEW:
            filtered_tokens_short = filtered_tokens_list[:NO_OF_WORDS_TAKEN_FROM_REVIEW]
        else:
            filtered_tokens_short = filtered_tokens_list
            filtered_tokens_short += (NO_OF_WORDS_TAKEN_FROM_REVIEW - filtered_tokens_len) * ['0']

        vectors_list = []
        for token in filtered_tokens_short:
            vector = _google_model[token]
            vectors_list.append(vector)

        vectors_matrix = np.zeros([NO_OF_WORDS_TAKEN_FROM_REVIEW, 300])
        for row in range(len(vectors_list)):
            vectors_matrix[row] = vectors_list[row]
        yield vectors_matrix, idx
        idx += 1


def main():
    read_csv()
    load_google_w2v_model()
    # counter = 0
    for review_vectors_matrix, review_idx in create_review_matrix(reviews):
        write_vectors_matrix_to_file(review_idx, review_vectors_matrix)
        # if counter < 10:
        #     write_vectors_matrix_to_file(review_idx, review_vectors_matrix)
        # else:
        #     break
        # counter += 1


if __name__ == '__main__':
    main()
