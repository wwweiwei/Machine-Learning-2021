import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import time
import argparse
import os
import os.path
from os import path

dim = 100

def load_image(filename):
    img = Image.open(filename)
    width, height = img.size
    pixel = np.array(img.getdata()).reshape((width*height, 3))

    coord = np.array([]).reshape(0, 2)
    for i in range(dim):
        row_x = np.full(dim, i)
        row_y = np.arange(dim)
        row = np.array(list(zip(row_x, row_y))).reshape(1*dim, 2)
        coord = np.vstack([coord, row])

    return pixel, coord

def initial(data, initial_method):
    '''
    1. random: random from existing data
    2. Kmeans++: push the centroids as far from one another as possible
    '''
    prev_classification = np.random.randint(K, size=data.shape[0])
    if initial_method == 'random':
        candidate = np.random.randint(low=0, high=data.shape[0], size=K)
        mu = np.zeros([K, K], dtype=np.float32)
        for i in range(K):
            mu[i, :] = data[candidate[i], :]
        return mu, prev_classification
    
    elif initial_method == 'Kmeans++':
        mu = np.zeros([K, K], dtype=np.float32)
        first_cluster = np.random.randint(low=0, high=data.shape[0], size=1, dtype=np.int)
        mu[0, :] = data[first_cluster, :]
        for i in range(1,K):
            distance = np.zeros(data.shape[0], dtype=np.float32)
            for j in range(0, data.shape[0]):
                distance[j] = np.linalg.norm(data[j, :] - mu[0, :])
            distance = distance / distance.sum()
            candidate = np.random.choice(data.shape[0], 1, p=distance)
            mu[i, :] = data[candidate, :]
        return mu, prev_classification

def compute_kernel(color, coord, gamma_s, gamma_c):
    '''
    self-defined kernel function that uses two RBF (spatial and color information)
    '''
    gram_matrix = np.zeros((len(color), len(color)))
    spatial_sq_dists = squareform(pdist(coord, 'sqeuclidean'))
    RBF_spatial = np.exp(-gamma_s*spatial_sq_dists)
    
    color_sq_dists = squareform(pdist(color, 'sqeuclidean'))
    RBF_color = np.exp(-gamma_c*color_sq_dists)

    return RBF_spatial * RBF_color

def classify(data, mu):
    classification = np.zeros(data.shape[0], dtype=np.int)
    for idx in range(data.shape[0]):
        distance = np.zeros(mu.shape[0], dtype=np.float)
        for cluster in range(mu.shape[0]):
            delta = abs(np.subtract(data[idx, :], mu[cluster, :]))
            distance[cluster] = np.square(delta).sum(axis=0)
        classification[idx] = np.argmin(distance)

    return classification

def MAE(classification, prev_classification):
    error = 0
    for i in range(classification.shape[0]):
        error += np.absolute(classification[i] - prev_classification[i])

    return error

def visualization(filename, storename, iteration, classification, initial_method):
    img = Image.open('./data/' + filename)
    width, height = img.size
    pixel = img.load()
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel[j, i] = color[classification[i * dim + j]]
    img.save(storename)

def show_eigenspace(storename, classification, data):
    color = iter(plt.cm.rainbow(np.linspace(0, 1, K)))
    plt.clf()
    plt.suptitle('Spectral-clustering in eigen-space')
    for cluster in range(K):
        col = next(color)
        for j in range(0, data.shape[0]):
            if classification[j] == cluster:
                plt.scatter(data[j][0], data[j][1], s=8, c=[col])
    plt.savefig(storename)

def update(data, mu, classification):
    new_mu = np.zeros(mu.shape, dtype=np.float32)
    count = np.zeros(mu.shape, dtype=np.int)
    one = np.ones(mu.shape[1], dtype=np.int)
    for idx in range(data.shape[0]):
        new_mu[classification[idx]] += data[idx]
        count[classification[idx]] += one
    for i in range(new_mu.shape[0]):
        if count[i][0] == 0:
            count[i] += one
    
    return np.true_divide(new_mu, count)

def K_Means(data, filename, ratio):
    method = ['random', 'Kmeans++']
    for initial_method in method:
        print('Initial method: {}'.format(initial_method))
        start_time = time.time()        
        mu, classification = initial(data, initial_method)
        # print('mu = {}'.format(mu))

        iteration = 0
        error = 0
        prev_error = 0
        while(iteration <= epochs):
            iteration += 1
            print('iteration = {}'.format(iteration))
            prev_classification = classification

            if ratio == False:
                path_ = 'spectral_result/' + str(filename) + '/' + 'K_' + str(K) + '/' + initial_method + '/'
            else:
                path_ = 'spectral_ratio_result/' + str(filename) + '/' + 'K_' + str(K) + '/' + initial_method + '/'

            if not path.exists(path_):
                os.makedirs(path_)
            storename = path_ + 'iter_' + str(iteration) + '_' + str(gamma_c) + '_' + str(gamma_s) +'.png'
             
            visualization(filename, storename, iteration, classification, initial_method)
            classification = classify(data, mu)
            error = MAE(classification, prev_classification)
            print('error = {}'.format(error))
            if error == prev_error:
                break
            prev_error = error
            mu = update(data, mu, classification)

        if ratio == False:
            path_ = 'eigenspace_result/' + str(filename) + '/' + 'K_' + str(K) + '/' + initial_method + '/'
        else:
            path_ = 'eigenspace_ratio_result/' + str(filename) + '/' + 'K_' + str(K) + '/' + initial_method + '/'
        
        if not path.exists(path_):
            os.makedirs(path_)
        eigenspace_storename = path_ + str(gamma_c) + '_' + str(gamma_s) + '_' + 'eigenspace' + '_'+ str(K) + '.png'
        
        show_eigenspace(eigenspace_storename, classification, data)        
        print('Execution time: %s seconds' % (time.time() - start_time))


def normalized_cut(pixel, coord, gamma_s, gamma_c):
    weight = compute_kernel(pixel, coord, gamma_s, gamma_c)
    degree = np.diag(np.sum(weight, axis=1))

    degree_square = np.diag(np.power(np.diag(degree), -0.5)) ## D**-0/5
    L_sym = np.eye(weight.shape[0]) - degree_square @ weight @ degree_square ## normalized Laplacian
    eigen_values, eigen_vectors = np.linalg.eig(L_sym)
    idx = np.argsort(eigen_values)[1: K+1] ## compute the first K eigenvectors
    U = eigen_vectors[:, idx].real.astype(np.float32)

    ## form the matrix from U by normalizing the rows to norm 1
    sum_over_row = (np.sum(np.power(U, 2), axis=1) ** 0.5).reshape(-1, 1)
    T = U.copy()
    for i in range(sum_over_row.shape[0]):
        if sum_over_row[i][0] == 0:
            sum_over_row[i][0] = 1
        T[i][0] /= sum_over_row[i][0]
        T[i][1] /= sum_over_row[i][0]

    ## return clusters
    return T

def ratio_cut(pixel, coord, gamma_s, gamma_c): ## unnormalized Laplacian
    weight = compute_kernel(pixel, coord, gamma_s, gamma_c)
    degree = np.diag(np.sum(weight, axis=1))
    ## L = D - W
    L = degree - weight

    eigen_values, eigen_vectors = np.linalg.eig(L)
    ## compute the first k generalized eigenvectors u1, .., uk of the generalized eigen prob
    idx = np.argsort(eigen_values)[1: K+1]
    U = eigen_vectors[:, idx].real.astype(np.float32)

    return U
    
if __name__ == '__main__':
    ## parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--K', type = int, default = 2)
    parser.add_argument('--gamma_c', type = float, default = 1 / (255*255))
    parser.add_argument('--gamma_s', type = float, default = 1 / (100*100))

    args = parser.parse_args()
    epochs = args.epochs
    K = args.K
    gamma_c = args.gamma_c
    gamma_s = args.gamma_s
    ratio = False

    filename = 'image1.png'
    pixel1, coord1 = load_image('./data/' + filename)
    T = normalized_cut(pixel1, coord1, gamma_s, gamma_c)
    K_Means(T, filename, ratio)
    
    filename = 'image2.png'
    pixel2, coord2 = load_image('./data/' + filename)
    T = normalized_cut(pixel2, coord2, gamma_s, gamma_c)
    K_Means(T, filename, ratio)

    ##################################
    print('------------------------')
    ratio = True

    filename = 'image1.png'
    pixel1, coord1 = load_image('./data/' + filename)
    U = ratio_cut(pixel1, coord1, gamma_s, gamma_c)
    K_Means(U, filename, ratio)

    filename = 'image2.png'
    pixel2, coord2 = load_image('./data/' + filename)
    U = ratio_cut(pixel2, coord2, gamma_s, gamma_c)
    K_Means(U, filename, ratio)