import math
import numpy as np
import os
import re
import random
import argparse
from PIL import Image
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

storedir = './PCA_Result/'
traindir = './Training/'
testdir = './Testing/'

def readInput(dir):
    files = os.listdir(dir)
    data = []
    target = []
    totalfile = []
    for file in files:
        '''
        E.g.
            file = 'subject01.leftlight.pgm'
            number = 01
        '''
        totalfile.append(file)
        filename = dir + file
        number = int(re.sub(r'\D', "", file.split('.')[0]))
        target.append(number)
        img = Image.open(filename)
        img = img.resize((60, 60), Image.ANTIALIAS)
        width, height = img.size
        pixel = np.array(img.getdata()).reshape((width*height))
        data.append(pixel)
    return np.array(data), np.array(target).reshape(-1,1), np.array(totalfile)

def computeEigen(A):
    '''
    linalg.eigh(): return the eigenvalues and eigenvectors 
    of a complex Hermitian (conjugate symmetric) or a real symmetric matrix
    '''
    eigen_values, eigen_vectors = np.linalg.eigh(A)

    print('eigen_values = {}'.format(eigen_values.shape))
    largest_idx = eigen_values.argsort()[::-1]
    return eigen_vectors[:,largest_idx][:,:25]

def PCA(data):
    covariance = np.cov(data.T)
    eigen_vectors = computeEigen(covariance)
    lower_dimension_data = np.matmul(data, eigen_vectors)

    print('eigen vector shape = {}'.format(eigen_vectors.shape))
    print('lower dimension data shape = {}'.format(lower_dimension_data.shape))
    return lower_dimension_data, eigen_vectors

def kernelPCA(data, gamma, kernel):
    lower_dimension_data = None
    if kernel == 'rbf':
        ## computation of the kernel (similarity) matrix.
        sq_dists = squareform(pdist(data), 'sqeuclidean')
        gram_matrix = np.exp(-gamma * sq_dists)
        N = gram_matrix.shape[0]
        one_n = np.ones((N, N)) / N ## 1/N
        ## eigendecomposition of the kernel matrix
        K = gram_matrix - one_n.dot(gram_matrix) - gram_matrix.dot(one_n) + one_n.dot(gram_matrix).dot(one_n)
        eigen_vectors = computeEigen(K)
        lower_dimension_data = np.matmul(gram_matrix, eigen_vectors)
    else: ## sigmoid
        data = np.tanh(data)
        gram_matrix = np.matmul(data, data.T)
        N = gram_matrix.shape[0]
        one_n = np.ones((N, N)) / N
        K = gram_matrix - one_n.dot(gram_matrix) - gram_matrix.dot(one_n) + one_n.dot(gram_matrix).dot(one_n)
        eigen_vectors = computeEigen(K)
        lower_dimension_data = np.matmul(gram_matrix, eigen_vectors)
    return lower_dimension_data

def visualization(dirname, totalfile, data):
    ## random pick 10 images
    randomIdx = random.sample(range(0, 135), 10)
    totalfile = totalfile[randomIdx]

    ## original image
    idx = 0
    for file in totalfile:
        plt.subplot(2, 5, idx+1)
        filename = dirname + file
        img = Image.open(filename)
        img = img.resize((60, 60), Image.ANTIALIAS)
        plt.imshow(img, plt.cm.gray)
        idx += 1
    plt.suptitle('Original')
    plt.savefig('PCA_original.png') 
    plt.show()

    ## reconstructive image
    for idx in range(0, 10):
        plt.subplot(2, 5, idx+1)
        pixel = data[randomIdx[idx]].reshape((60, 60)).copy()
        plt.imshow(pixel, plt.cm.gray)
    plt.suptitle('Reconstruction')
    plt.savefig('PCA_reconstruction.png')
    plt.show()

def drawEigenface(storedir, eigen_vectors):
    eigen_vectors = eigen_vectors.T
    for i in range(0, 25):
        plt.subplot(5, 5, i+1)
        plt.imshow(eigen_vectors[i].reshape((60, 60)), plt.cm.gray)
        if i == 24:
            plt.savefig(storedir + 'PCA_Eigen-Face.png')
    plt.suptitle('PCA eigenfaces')
    plt.show()

def KNN(traindata, testdata, target):
    '''
    Calculate the distance between each training data and testing data.
    Pick the closest training data and consider the label as target label for each testing data.
    '''
    result = np.zeros(testdata.shape[0])
    for testIdx in range(testdata.shape[0]):
        distance_matrix = np.zeros(traindata.shape[0])
        for trainIdx in range(traindata.shape[0]):
            distance_matrix[trainIdx] = np.sqrt(np.sum((testdata[testIdx] - traindata[trainIdx]) ** 2))
        result[testIdx] = target[np.argmin(distance_matrix)]
    print('Result: ', result)
    print('distance_matrix size: ', distance_matrix.shape)
    return result

def checkAccuracy(target, predict):
    correct = 0
    for i in range(len(target)):
        if target[i] == predict[i]:
            correct += 1
    print('Accuracy = {:.3f}  ({} / {})'.format(correct / len(target), correct, len(target)))

if __name__ == '__main__':
    ## parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, default = 'PCA')
    parser.add_argument('--gamma', type = float, default = 1e-6, help = 'just for kernel PCA')
    parser.add_argument('--kernel', type = str, default = 'rbf', help = 'just for kernel PCA')


    args = parser.parse_args()
    mode = args.mode
    gamma = args.gamma
    kernel = args.kernel

    training_data, training_target, training_totalfile = readInput(traindir)
    testing_data, testing_target, testing_totalfile = readInput(testdir)
    print('input data shape = {}'.format(training_data.shape))

    if mode == 'PCA':
        storefolder = '/PCA/'
        lower_dimension_data, eigen_vectors = PCA(training_data)

        reconstruct_data = np.matmul(lower_dimension_data, eigen_vectors.T)
        print('reconstruct data shape = {}'.format(reconstruct_data.shape))

        visualization(traindir, training_totalfile, reconstruct_data)
        drawEigenface(storedir + storefolder, eigen_vectors)

        ## face recognition
        data = np.concatenate((training_data, testing_data), axis=0)
        lower_dimension_data, eigen_vectors = PCA(data)

    else: ## kernel PCA
        storefolder = '/kernelPCA/'

        ## face recognition
        data = np.concatenate((training_data, testing_data), axis=0)
        lower_dimension_data = kernelPCA(data, gamma, kernel)

    lower_dimension_testing_data = lower_dimension_data[training_totalfile.shape[0]:].copy()
    lower_dimension_data = lower_dimension_data[:training_totalfile.shape[0]].copy()
    # print('lower dimension data (train) shape: {}'.format(lower_dimension_data.shape))
    # print('lower dimension data (test) shape: {}'.format(lower_dimension_testing_data.shape))

    predict = KNN(lower_dimension_data, lower_dimension_testing_data, training_target)
    checkAccuracy(testing_target, predict)