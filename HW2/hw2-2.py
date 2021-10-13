import argparse
import csv
from collections import Counter
import math
'''
python hw2-2.py --INPUT_FILE=testfile.txt --A=0 --B=0
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_FILE', type = str, default = 'testfile.txt')
    parser.add_argument('--A', type = int, default = 0)
    parser.add_argument('--B', type = int, default = 0)
    args = parser.parse_args()
    
    INPUT_FILE = args.INPUT_FILE
    a = args.A
    b = args.B

    data = []
    data_count = []

    with open(INPUT_FILE, 'r') as file:
        file_rows = csv.reader(file, delimiter = ',')
        for idx, row in enumerate(file_rows):
            data.append(row[0])
            row = list(row[0])
            row_count = Counter(row)
            data_count.append([row_count['0'], row_count['1']])

    '''
    N = a+b
    m = b
    Binomial_MLE = m/N
    likelihood = C{Nm} P^m (1-P)^(N-m)
    '''

    for row in range(len(data)):
        print('case {} : '.format(row + 1), data[row])
        
        N = data_count[row][0] + data_count[row][1]
        m = data_count[row][1]
        Binomial_MLE = m/N
        likelihood = math.factorial(N) / (math.factorial(m) * math.factorial(N-m)) * (Binomial_MLE ** m) * ((1 - Binomial_MLE) ** (N-m))
        
        print('Likelihood : {}'.format(likelihood))
        print('Beta prior :     a = {a}, b = {b}'.format(a=a,b=b))

        a += data_count[row][0]
        b += data_count[row][1]

        print('Beta posterior : a = {a}, b = {b}'.format(a=a,b=b), '\n')
