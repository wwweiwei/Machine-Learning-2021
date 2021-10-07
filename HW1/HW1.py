import os
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision = 3, suppress=True)

''' 
python hw1.py --INPUT_FILE=testfile.txt --N=3 --LAMBDA=10000
''' 

def matrix_A(x, base):
    A = np.zeros((len(x), base))
    for i in range(len(x)):
        for j in range(base-1, -1, -1):
            A[i][base-j-1] = np.power(x[i], j)
    return A


def LU_decomposition(A):
    ''' 
    the elementary row operations must be performed from top to bottom within each column
    and column by column from left to right
    return
        L: lower triangular matrix
        A: upper triangular matrix
    ''' 
    row_size, col_size = np.shape(A)

    L = np.identity(row_size)
    for row in range(row_size):
        L_tmp = np.identity(row_size)
        for col in range(col_size):
            if col < row and A[row][col] != 0:
                L_tmp[row][col] = (-1) * A[row][col] / A[col][col]
                A = np.matmul(L_tmp, A)
                L[row][col] = (-1) * L_tmp[row][col]
                L_tmp = np.identity(row_size)
    return L, A


def forward_substitution(L, b):
    '''
    Ly = b, solve y
    '''
    y = [b_value for b_value in b]
    for row in range(len(L)):
        for col in range(row):
            y[row] -= y[col] * L[row][col]
        y[row] /= L[row][row]
    return y

def backward_substitution(U, y):
    '''
    Ux = y, solve x
    '''
    x = [y_value for y_value in y]
    for row in range(len(U)-1,-1,-1):
        for col in range(len(U)-1,row,-1):
            x[row] -= x[col] * U[row][col]
        x[row] /= U[row][row]
    return x

def substitution(L, U, b):
    '''
    return
        (LU)_inverse * A_transpose * b
        = U_inverse * L_inverse * A_transpose * b
    '''
    # Ly = b, solve y
    y = forward_substitution(L, b)

    # Ux = y, solve x
    x = backward_substitution(U, y)

    return x

def plot(model_name, x, y, b):
    fig = plt.figure()
    plt.title(model_name)
    plt.plot(x, y, 'ro')
    plt.plot(x, b, '-k')
    plt.show()
    fig.savefig(model_name + '.png')

def show_result(A, x, y, x1, model_name):
    ''' 
    A: data, x: parameter, b: target
    b = Ax
    '''
    b = np.matmul(A,x)
    print(model_name + ': ')
    
    print('Fitting line : ', end='')
    for row in range(np.shape(x)[0]):
        if row != np.shape(x)[0] - 1:
            if row != 0:
                if x[row] >= 0:
                    print('+ ' + str(x[row][0]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
                else:
                    print('- ' + str((-1) * x[row][0]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
            else:
                if x[row] >= 0:
                    print(str(x[row][0]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
                else:
                    print('- ' + str((-1) * x[row][0]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
        else:
            if x[row] >= 0:
                print('+ ' + str(x[row][0]))
            else:
                print('- ' + str((-1) * x[row][0]))

    # Calculate total error
    total_error = 0
    for i in range(np.shape(b)[0]):
        total_error += np.square(b[i] - y[i])
    print('Total error: ', total_error[0])

    plot(model_name, x1, y, b)

def getGradient(A_transpose_A, x, A_transpose_b):
    '''
    Gradient = 2 * A_transpose * A * x - 2 * A_transpose * b
    '''
    A_transpose_A_x_2 = 2 * (np.matmul(A_transpose_A, x))
    A_transpose_b_2 = 2 * A_transpose_b
    gradient = np.subtract(A_transpose_A_x_2, A_transpose_b_2)
    return gradient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--FILENAME', type = str, default = 'testfile.txt')
    parser.add_argument('--N', type = int, default = 3)
    parser.add_argument('--LAMBDA', type = int, default = 10000)
    args = parser.parse_args()

    INPUT_FILE = args.FILENAME
    N = args.N
    LAMBDA = args.LAMBDA

    point = []
    with open(INPUT_FILE, 'r') as file:
        file_rows = csv.reader(file, delimiter = ',')
        for idx, row in enumerate(file_rows):
            point.append([float(row[0]), float(row[1])])
    data = np.array(point)

    ''' 
    gram_matrix: A_transpose * A
    ''' 
    x = data[:,0]
    b = data[:,1].reshape((len(data[:,1]), 1))
    A = matrix_A(data[:,0], N)
    gram_matrix = np.matmul(A.T, A)
    A_transpose_b = np.matmul(A.T, b)

    ''' 
    LSE
    rLSE = (A_transpose * A + lambda * I)^-1 A_transpose * b
    ''' 
    lambda_I = LAMBDA * np.identity(np.shape(gram_matrix)[0])
    gram_matrix_add_lambda_I = np.add(gram_matrix, lambda_I)
    L, U = LU_decomposition(gram_matrix_add_lambda_I)
    lse_result = substitution(L, U, A_transpose_b)

    show_result(A, lse_result, b, x, 'LSE')

    print('\n', end='')

    ''' 
    Newton's method
    Xn+1 = Xn - [H]^-1 * gradient
    Gradient = 2 * A_transpose * A * x - 2 * A_transpose * b
    Hession = 2 * A_transpose * A
    ''' 
    newton_guess = np.zeros((N ,1), dtype=float)
    A_transpose_b = np.matmul(A.T, b)
    gradient = getGradient(gram_matrix, newton_guess, A_transpose_b)

    hession = 2 * gram_matrix
    L, U = LU_decomposition(hession)
    hession_inv = np.zeros((N ,N), dtype=float)
    I_matrix = np.identity(N)
    for col in range(N):
        hession_inv[:,col] = substitution(L, U, I_matrix[:,col])
    newton_result = np.subtract(newton_guess, np.matmul(hession_inv, gradient))

    show_result(A, newton_result, b, x, 'Newton\'s Method')
