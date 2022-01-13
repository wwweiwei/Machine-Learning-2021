import numpy as np
import os
import re
import random
import argparse
from PIL import Image
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt

storedir = './LDA_result/'
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

def computeMean(data, target, num_class, num_subject):
    class_mean = np.zeros([num_class, data.shape[1]])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            class_mean[target[i][0]-1][j] += data[i][j]
    for i in range(num_class):
        for j in range(data.shape[1]):
            class_mean[i][j] /= num_subject
    total_mean = np.mean(data, axis=0).reshape(-1, 1)
    print('class_mean shape = ', class_mean.shape)
    print('total_mean shape = ', total_mean.shape)
    return class_mean, total_mean

def computeWithinClass(data, target, class_mean):
    '''
    SW = sum((mj-m)(mj-m)^T)
    size: 3600x3600
    '''
    within_class = np.zeros([data.shape[1], data.shape[1]])
    for i in range(data.shape[0]):
        dist = np.subtract(data[i], class_mean[target[i][0]-1]).reshape(data.shape[1], 1)
        within_class += np.matmul(dist, dist.T)
    print('within_class shape = ', within_class.shape)
    return within_class

def computeBetweenClass(class_mean, total_mean, num_class, num_subject):
    '''
    SB = sum(nj(mj-m)(mj-m)^T)
    size: 3600x3600
    '''
    betweenclass = np.zeros([data.shape[1], data.shape[1]])
    for i in range(num_class):
        dist = np.subtract(class_mean[i], total_mean[i]).reshape(data.shape[1], 1)
        betweenclass += np.matmul(dist, dist.T)
    betweenclass *= num_subject
    print('betweenclass shape = ', betweenclass.shape)
    return betweenclass

def computeEigen(A):
    '''
    linalg.eigh(): return the eigenvalues and eigenvectors 
    of a complex Hermitian (conjugate symmetric) or a real symmetric matrix
    '''
    eigen_values, eigen_vectors = np.linalg.eigh(A)
    print('eigen_values = {}'.format(eigen_values.shape))
    largest_idx = eigen_values.argsort()[::-1]
    return eigen_vectors[:,largest_idx][:,:25]

def kernelLDA(data, target, num_class, num_subject, gamma, kernel):
    '''
    numpy.linalg.pinv(): Compute the pseudo-inverse of a matrix
    '''
    if kernel == 'rbf':
        sq_dists = squareform(pdist(data), 'sqeuclidean')
        gram_matrix = np.exp(-gamma * sq_dists)
    else:
        data = np.tanh(data)
        gram_matrix = np.matmul(data, data.T)

    within_class = np.zeros([data.shape[0], data.shape[0]])
    I_minus_one = np.identity(num_subject) - (num_subject * np.ones((num_subject, num_subject)))
    for i in range(num_class):
        Kj = gram_matrix[np.where(target == i+1)[0]].copy()
        multiply = np.matmul(Kj.T, np.matmul(I_minus_one, Kj))
        within_class += multiply

    between_class = np.zeros([data.shape[0], data.shape[0]])
    for i in range(num_class):
        class_mean = gram_matrix[np.where(target == i+1)[0]].copy()
        class_mean = np.sum(class_mean, axis=0).reshape(-1, 1) / num_subject
        total_mean = gram_matrix[np.where(target == i+1)[0]].copy()
        total_mean = np.sum(total_mean, axis=0).reshape(-1, 1) / data.shape[0]
        dist = np.subtract(class_mean, total_mean)
        between_class += num_subject * np.matmul(dist, dist.T)

    eigenvectors = computeEigen(np.matmul(np.linalg.pinv(within_class), between_class))
    lower_dimension_data = np.matmul(gram_matrix, eigenvectors)
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
    plt.savefig('LDA_original.png') 
    plt.show()

    ## reconstructive image
    for idx in range(0, 10):
        plt.subplot(2, 5, idx+1)
        pixel = data[randomIdx[idx]].reshape((60, 60)).copy()
        plt.imshow(pixel, plt.cm.gray)
    plt.suptitle('Reconstruction')
    plt.savefig('LDA_reconstruction.png')
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
        alldist = np.zeros(traindata.shape[0])
        for trainIdx in range(traindata.shape[0]):
            alldist[trainIdx] = np.sqrt(np.sum((testdata[testIdx] - traindata[trainIdx]) ** 2))
        result[testIdx] = target[np.argmin(alldist)]
    print('result: ', result)
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
    parser.add_argument('--mode', type = str, default = 'LDA')
    parser.add_argument('--gamma', type = float, default = 1e-3, help = 'just for kernel LDA')
    parser.add_argument('--num_class', type = int, default = 15)
    parser.add_argument('--subject', type = int, default = 11)
    parser.add_argument('--kernel', type = str, default = 'rbf', help = 'just for kernel PCA')


    args = parser.parse_args()
    mode = args.mode
    gamma = args.gamma
    num_class = args.num_class
    subject = args.subject
    kernel = args.kernel

    training_data, training_target, training_totalfile = readInput(traindir)
    testing_data, testing_target, testing_totalfile = readInput(testdir)
    data = np.concatenate((training_data, testing_data), axis=0)
    target = np.concatenate((training_target, testing_target), axis=0)
    print('input data shape = {}, target shape = {}'.format(data.shape, target.shape))
    
    if mode == 'LDA':
        class_mean, total_mean = computeMean(data, target, num_class, subject)
        withinclass = computeWithinClass(data, target, class_mean)
        betweenclass = computeBetweenClass(class_mean, total_mean, num_class, subject)
        ## maximize between-class scatter and minimize within-class scatter
        eigenvectors = computeEigen(np.matmul(np.linalg.pinv(withinclass), betweenclass))

        lower_dimension_data = np.matmul(data, eigenvectors)
        lower_dimension_training_data = lower_dimension_data[:training_totalfile.shape[0]].copy()
        lower_dimension_testing_data = lower_dimension_data[training_totalfile.shape[0]:].copy()
        targettrain = target[:training_totalfile.shape[0]].copy()
        targettest = target[training_totalfile.shape[0]:].copy()
        
        reconstruct_data = np.matmul(lower_dimension_training_data, eigenvectors.T)
        visualization(traindir, training_totalfile, reconstruct_data)
        drawEigenface(storedir, eigenvectors)
    
    else: ## Kernel LDA
        lower_dimension_data = kernelLDA(data, target, num_class, subject, gamma, kernel)
        lower_dimension_training_data = lower_dimension_data[:training_totalfile.shape[0]].copy()
        lower_dimension_testing_data = lower_dimension_data[training_totalfile.shape[0]:].copy()
        targettrain = target[:training_totalfile.shape[0]].copy()
        targettest = target[training_totalfile.shape[0]:].copy()

    predict = KNN(lower_dimension_training_data, lower_dimension_testing_data, targettrain)
    checkAccuracy(targettest, predict)