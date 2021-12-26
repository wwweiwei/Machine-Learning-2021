import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist, squareform
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

def initial(data, initial_method, K):
    C_x = np.random.randint(0, dim, size=K)
    C_y = np.random.randint(0, dim, size=K)
    mu = np.random.randn(K, 2)

    if initial_method == 'random':
        prev_classification = np.random.randint(K, size=data.shape[0])
        return mu, prev_classification

    elif initial_method == 'equal-split':
        prev_classification = []
        border = dim * dim / K
        for i in range(data.shape[0]):
            prev_classification.append(int(i/border))
        prev_classification = np.asarray(prev_classification)
        return mu, prev_classification  

    elif initial_method == 'modK':
        prev_classification = []
        for i in range(data.shape[0]):
            prev_classification.append(i%K)
        prev_classification = np.asarray(prev_classification)
        return mu, prev_classification

     

def compute_kernel(color, coord, gamma_s, gamma_c):
    '''
    self-defined kernel function that uses two RBF (spatial and color information)
    '''
    spatial_sq_dists = squareform(pdist(coord, 'sqeuclidean'))
    RBF_spatial = np.exp(-gamma_s*spatial_sq_dists)

    color_sq_dists = squareform(pdist(color, 'sqeuclidean'))
    RBF_color = np.exp(-gamma_c*color_sq_dists)

    ## different kernel: polynomial
    # return spatial_sq_dists+color_sq_dists
    return RBF_spatial * RBF_color

def compute_third_term(kernel_data, classification, K):
    cluster_sum = np.zeros(K, dtype=np.int)
    kernel_sum = np.zeros(K, dtype=np.float)

    for i in range(classification.shape[0]):
        cluster_sum[classification[i]] += 1

    for cluster in range(K):
        for p in range(kernel_data.shape[0]):
            for q in range(kernel_data.shape[0]):
                if classification[p] == cluster and classification[q] == cluster:
                    kernel_sum[cluster] += kernel_data[p][q]

    for cluster in range(K):
        if cluster_sum[cluster] == 0:
            cluster_sum[cluster] = 1
        kernel_sum[cluster] /= (cluster_sum[cluster]**2)
    
    return kernel_sum

def compute_second_term(kernel_data, classification, idx, cluster):
    cluster_sum = 0
    kernel_sum = 0
    
    for i in range(classification.shape[0]):
        if classification[i] == cluster:
            cluster_sum += 1
    if cluster_sum == 0:
        cluster_sum = 1

    for i in range(kernel_data.shape[0]):
        if classification[i] == cluster: ## indicator = 1
            kernel_sum += kernel_data[idx][i]

    return (-2) * kernel_sum / cluster_sum

def classify(data, kernel_data, mu, classification, K):
    current_classification = np.zeros(data.shape[0], dtype=np.int)
    third_term = compute_third_term(kernel_data, classification, K)
    
    for idx in range(data.shape[0]):
        distance = np.zeros(K, dtype=np.float32)
        for cluster in range(K):
            distance[cluster] = compute_second_term(kernel_data, classification, idx, cluster) + third_term[cluster]
        current_classification[idx] = np.argmin(distance)

    return current_classification

def MAE(classification, prev_classification):
    error = 0
    for i in range(classification.shape[0]):
        error += np.absolute(classification[i] - prev_classification[i])

    return error

def visualization(filename, storename, classification):
    img = Image.open('./data/' + filename)
    width, height = img.size
    pixel = img.load()
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel[j, i] = color[classification[i * dim + j]]
    img.save(storename)

def kernel_k_means(filename, data, coord, epochs, K, gamma_c, gamma_s):
    method = ['random', 'equal-split', 'modK']

    for initial_method in method:
        print('Initial method: {}'.format(initial_method))    
        start_time = time.time()
        mu, classification = initial(data, initial_method, K)
        kernel_data = compute_kernel(data, coord, gamma_s, gamma_c)
        
        error = 0
        prev_error = 0
        iteration = 0
        while (iteration <= epochs):
            iteration += 1
            print('iteration = {}'.format(iteration))
            prev_classification = classification

            path_ = 'result/' + str(filename) + '/' + 'K_' + str(K) + '/' + initial_method + '/'
            if not path.exists(path_):
                os.makedirs(path_)
            storename = path_ + 'iter_' + str(iteration) + '_' + str(gamma_c) + '_' + str(gamma_s) +'.png'
            
            visualization(filename, storename, classification)
            classification = classify(data, kernel_data, mu, classification, K)
            error = MAE(classification, prev_classification)
            print('error = {}'.format(error))

            if error == prev_error:
                break
            prev_error = error
        print('Execution time: %s seconds' % (time.time() - start_time))


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

    filename = 'image1.png'
    pixel1, coord1 = load_image('./data/' + filename)
    kernel_k_means(filename, pixel1, coord1, epochs, K, gamma_c, gamma_s)

    filename = 'image2.png'
    pixel2, coord2 = load_image('./data/' + filename)
    kernel_k_means(filename, pixel2, coord2, epochs, K, gamma_c, gamma_s)
