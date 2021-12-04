import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import cdist

def read_file():
    X_train = np.loadtxt('./data/X_train.csv', dtype=np.float, delimiter=',')
    Y_train = np.loadtxt('./data/Y_train.csv', dtype=np.float, delimiter=',')
    X_test = np.loadtxt('./data/X_test.csv', dtype=np.float, delimiter=',')
    Y_test = np.loadtxt('./data/Y_test.csv', dtype=np.float, delimiter=',')
    return X_train, Y_train, X_test, Y_test

def different_kernel_func(X_train, Y_train, X_test, Y_test):
    '''
    svm parameter
        -t: type of kernel function
            0: linear, 1:polynomial, 2:RBF, 3: sigmoid
    '''
    kernel = ['linear', 'polynomial', 'RBF']
    for i in range(3):
        print('Kernel function: {}'.format(kernel[i]))
        parameter = '-q -t ' + str(i)
        prob  = svm_problem(Y_train, X_train)
        param = svm_parameter(parameter)
        model = svm_train(prob, param)
        svm_predict(Y_test, X_test, model)

def grid_search(X_train, Y_train, X_test, Y_test):
    '''
    svm parameter
        -v: cross validation
        -q: silent mode
        -t: type of kernel function
            0: linear, 1:polynomial, 2:RBF, 3: sigmoid, 4: precomputed
        -c: cost(set C-SVC) 
    '''
    cost = ['0.25', '0.5', '1', '2', '4']
    gamma = ['0.25', '0.5', '1', '2', '4']
    degree = ['2', '3', '4']
    coef0 = ['0', '1', '2']
    best_parameter = ''
    best_accuracy = 0
    
    prob  = svm_problem(Y_train, X_train)
    
    ## linear kernel
    print('[Linear kernel]')
    for c in cost:
        ## 3-fold cross validation
        parameter = '-v 3 -q -t 0 -c ' + c
        param = svm_parameter(parameter)
        accuracy = svm_train(prob, param)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameter = parameter
    
    print('Best accuracy: {}'.format(best_accuracy))
    print('Best parameters: {}'.format(best_parameter))
    best_parameter = ''
    best_accuracy = 0
    
    ## polynomial kernel
    print('[Polynomial kernel]')
    for c in cost:
        for g in gamma:
            for d in degree:
                for r in coef0:
                    parameter = '-v 3 -q -t 1 -c ' + c + ' -g ' + g + ' -d ' + d + ' -r ' + r
                    param = svm_parameter(parameter)
                    accuracy = svm_train(prob, param)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_parameter = parameter
                        
    print('Best accuracy: {}'.format(best_accuracy))
    print('Best parameters: {}'.format(best_parameter))
    best_parameter = ''
    best_accuracy = 0
    
    ## RBF kernel
    print('[RBF kernel]')
    for c in cost:
        for g in gamma:
            parameter = '-v 3 -q -t 2 -c ' + c + ' -g ' + g
            param = svm_parameter(parameter)
            accuracy = svm_train(prob, param)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_parameter = parameter
    
    print('Best accuracy: {}'.format(best_accuracy))
    print('Best parameters: {}'.format(best_parameter))


def user_defined_kernel(X_train, Y_train, X_test, Y_test):
    gamma = 1/4
    train_linear_kernel = X_train.dot(X_train.transpose())
    train_rbf_kernel = np.exp(-gamma * cdist(X_train, X_train, 'sqeuclidean'))
    X_train_kernel = np.concatenate((np.arange(1, 5001).reshape((5000, 1)), train_linear_kernel + train_rbf_kernel), axis=1)
    
    test_linear_kernel = X_test.dot(X_train.transpose())
    test_rbf_kernel = np.exp(-gamma * cdist(X_test, X_train, 'sqeuclidean'))
    X_test_kernel = np.concatenate((np.arange(1, 2501).reshape((2500, 1)), test_linear_kernel + test_rbf_kernel), axis=1)

    prob  = svm_problem(Y_train, X_train_kernel, isKernel=True)
    param = svm_parameter('-q -t 4')
    model = svm_train(prob, param)
    svm_predict(Y_test, X_test_kernel, model)
    
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = read_file()

    print('Part 1 ---------------')
    different_kernel_func(X_train, Y_train, X_test, Y_test)
    print('Part 2 ---------------')
    grid_search(X_train, Y_train, X_test, Y_test)
    print('Part 3 ---------------')
    user_defined_kernel(X_train, Y_train, X_test, Y_test)
