import argparse
import csv
from collections import Counter
import math

'''
python hw2-3.py --INPUT_FILE=testfile.txt --A=0 --B=0
'''

def beta(p, a, b):
    prob = (p **(a-1)) * ((1-p) **(b-1))
    gamma = math.gamma(a+b) / ( math.gamma(a) * math.gamma(b) )
    return prob*gamma

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


    for row in range(1):        
        N = data_count[row][0] + data_count[row][1]
        m = data_count[row][1]

        p = b/(a+b)
        prior = (p**(b-1)) * ((1-p)**(a-1)) * ( (math.gamma(a) * math.gamma(b)) / math.gamma(a+b))
        likelihood = (math.factorial(N) / (math.factorial(m) * math.factorial(N-m))) * (p ** m) * ((1 - p) ** (N-m))
        marginal = (math.factorial(N) / (math.factorial(m) * math.factorial(N-m))) * ((math.gamma(a) * math.gamma(b)) / math.gamma(a+b)) * ( (math.gamma(m+a) * math.gamma(N-m+b)) / math.gamma(a+N+b))
   
        print('Beta prior :     a = {a}, b = {b}'.format(a=a,b=b))

        a += data_count[row][0]
        b += data_count[row][1]

        print('Beta posterior : a = {a}, b = {b}'.format(a=a,b=b), '\n')
        posterior_beta_pdf = beta(b/(a+b), a, b)
        print('Posterior (beta distribution): ', posterior_beta_pdf)

        '''
        posterior = (likelihood * prior) / marginal
        '''
        posterior = (likelihood * prior) / marginal
        print('Posterior ((likelihood * prior) / marginal): ', posterior, '\n')

        ## compare posterior and beta distribution
