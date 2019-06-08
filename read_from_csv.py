import csv

recommendations = []
reviews = []


def read_csv():
    with open('resources/steam_reviews.csv') as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        headers = []
        headers_read = False

        line_no = 0
        for row in read_csv:
            print(row)
            if not headers_read:
                headers = row
                headers_read = True
                continue
            print('===========')
            print('Line: %d' % line_no)
            print('Recomendation %s' % row[headers.index('recommendation')])
            recommendations.append(row[headers.index('recommendation')])
            print('Review %s' % row[headers.index('review')])
            reviews.append(row[headers.index('review')])
            print('===========')
            line_no += 1
        print("END")


def main():
    read_csv()


if __name__ == '__main__':
    main()
