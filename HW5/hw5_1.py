import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def read_file():
    data_x = []
    data_y = []
    with open('./data/input.data') as file:
        for line in file:
            data_x.append(float(line.split()[0]))
            data_y.append(float(line.split()[1]))
    train_x = np.array(data_x).reshape(-1, 1)
    train_y = np.array(data_y).reshape(-1, 1)
    return train_x, train_y

def kernel(X1, X2, params):
    '''
    Rational Quadratic Kernel
    lengthscale => l
    amplitude => sigma
    scale-mixture => alpha
    '''
    l = params[0]
    sigma = params[1]
    alpha = params[2]
    return (sigma ** 2) * (1 + (cdist(X1, X2, 'sqeuclidean') / 2 * alpha * (l ** 2))) ** (-alpha)

def predict(train_x, params):
    mu = np.zeros(train_x.shape)
    cov = kernel(train_x, train_x, params) + beta_inv * np.identity(train_x.shape[0])
    cov_inv = np.linalg.inv(cov)
    return mu, cov_inv

def add_noise(train_x, train_y, params, test_num, mu, cov_inv):
    test_x = np.linspace(-60, 60, test_num).reshape(-1, 1)
    test_y = np.empty(test_num).reshape(-1, 1)
    test_y_plus = np.empty(test_num).reshape(-1, 1)
    test_y_minus = np.empty(test_num).reshape(-1, 1)
    
    k_test = kernel(test_x, test_x, params) + beta_inv
    k_train_test = kernel(train_x, test_x, params)
    test_y = np.linalg.multi_dot([k_train_test.T, cov_inv, train_y])
    std = np.sqrt(k_test - np.linalg.multi_dot([k_train_test.T, cov_inv, k_train_test]))
    ## 95% CI => +- 2*std
    test_y_plus = test_y + 2 * (np.diag(std).reshape(-1, 1))
    test_y_minus = test_y - 2 * (np.diag(std).reshape(-1, 1))

    return test_x, test_y, test_y_minus, test_y_plus

def negative_log_likelihood(params, train_x, train_y):
    C = kernel(train_x, train_x, params)
    N = train_x.shape[0]
    item1 = -0.5 * np.log(np.linalg.det(C))
    item2 = -0.5 * np.linalg.multi_dot([train_y.T, np.linalg.inv(C), train_y])
    item3 = -0.5 * N * np.log(2*np.pi)
    negative_log_likelihood = - (item1 + item2 + item3)
    return negative_log_likelihood[0][0]

def draw(train_x, train_y, test_x, test_y, test_y_minus, test_y_plus, params, mode):
    fig = plt.figure()
    plt.title('Scalelength : {0:.2f}, Sigma : {1:.2f}, Alpha : {2:.2f}'.format(params[0], params[1], params[2]))
    plt.fill_between(test_x.ravel(), test_y_plus.ravel(), test_y_minus.ravel(), facecolor = 'cyan', label = '95% CI')
    plt.scatter(train_x, train_y, color = 'blue', label = 'data')
    plt.plot(test_x.ravel(), test_y.ravel(), color = 'black', label = 'prediction')
    plt.xlim(-60, 60)
    plt.legend(loc = 'upper right', frameon = True) 
    plt.show()
    fig.savefig('gaussian_process_' + mode + '.png')

if __name__ == '__main__':
    train_x, train_y = read_file()
    ## params = [lengthscale, amplitude, scale-mixture]
    params = [1.0, 1.0, 1.0]
    test_num = 120
    beta_inv = 1/5

    ## --- Part 1 ---
    ## gaussian process as a prior defined by the kernel function
    mu, cov_inv = predict(train_x, params)
    test_x, test_y, test_y_minus, test_y_plus = add_noise(train_x, train_y, params, test_num, mu, cov_inv)
    draw(train_x, train_y, test_x, test_y, test_y_minus, test_y_plus, params, 'part1')


    ## --- Part 2 ---
    ## optimize the kernel parameters by minimizing negative marginal log-likelihood
    min = minimize(fun = negative_log_likelihood, x0 = params, args = (train_x, train_y), bounds = ((1e-3, None), (1e-3, None), (1e-3, None)), method = 'L-BFGS-B', options = {})

    ## optimize result
    mu, cov_inv = predict(train_x, min.x)
    test_x, test_y, test_y_minus, test_y_plus = add_noise(train_x, train_y, min.x, test_num, mu, cov_inv)
    draw(train_x, train_y, test_x, test_y, test_y_minus, test_y_plus, min.x, 'part2')