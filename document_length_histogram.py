import csv
import re
import matplotlib.pyplot as plt

review_no = []
words_no_in_reviews = []


def read_reviews_from_csv():
    with open('resources/steam_reviews.csv') as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        # row_count = sum(1 for row in read_csv)  # fileObject is your csv.reader
        # print(row_count)
        headers = []
        headers_read = False
        review_idx = 0
        for row in read_csv:
            if not headers_read:
                headers = row
                headers_read = True
                continue
            review_no.append(review_idx)
            review_words_list = re.split('\\. | ', row[headers.index('review')])
            words_no_in_reviews.append(len(review_words_list))
            review_idx += 1

def show_reviews_box_plot(data):
    # spread = data
    # center = mean(data)
    data_median = median(data)
    sub_data_Q1 = list(filter(lambda x: x < data_median, data))
    sub_data_Q3 = list(filter(lambda x: x > data_median, data))
    Q1 = median(sub_data_Q1)
    Q3 = median(sub_data_Q3)
    IQR = Q3 - Q1
    flier_low = max(min(data), Q1 - 1.5 * IQR)
    flier_high = min(max(data), Q3 + 1.5 * IQR)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Number of words in reviews box plot')
    ax1.boxplot(data, showfliers=False)
    plt.hlines(y=[flier_high, mean(data), Q1, Q3, data_median, flier_low], xmin=0, xmax=1, colors='k', linestyles='dashed')
    plt.yticks([flier_high, Q1, Q3, mean(data), data_median, flier_low])
    plt.ylabel("word count")
    plt.show()


def show_reviews_length_histogram(bin_count=350):
    plt.title("Reviews length histogram")
    plt.xlabel("reviews count")
    plt.ylabel("word count")
    plt.hist(words_no_in_reviews, bins=bin_count, range=(1, 350))
    plt.show()


def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n


def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss


def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5


def median(lst):
    n = len(lst)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(lst)[n//2]
    else:
            return sum(sorted(lst)[n//2-1:n//2+1])/2.0


def main():
    read_reviews_from_csv()
    counter = 0
    for words in words_no_in_reviews:
        if words > 45:
            counter += 1
    print("Max review length is: %d" % max(words_no_in_reviews))
    print("Number of reviews with more than 45 words: %d" % counter)
    print("Average length of review is %f" % mean(words_no_in_reviews))
    print("Median of number of words in reviews is %d" % median(words_no_in_reviews))
    print("Standard deviation of review length is %f" % stddev(words_no_in_reviews))
    show_reviews_length_histogram()
    show_reviews_box_plot(words_no_in_reviews)


if __name__ == '__main__':
    main()