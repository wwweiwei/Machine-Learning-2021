import numpy as np
import numba as nb
import argparse
import warnings
warnings.filterwarnings('ignore')

'''
python hw4_2.py
'''

def open_file():
	data_type = np.dtype('int32').newbyteorder('>')
	
	data = np.fromfile('./train-images.idx3-ubyte', dtype = 'ubyte')
	train_image = data[4 * data_type.itemsize:].astype('float64').reshape(60000, 28 * 28)
	train_image_bin = np.divide(train_image, 128).astype('int')

	train_label = np.fromfile('./train-labels.idx1-ubyte',dtype = 'ubyte').astype('int')
	train_label = train_label[2 * data_type.itemsize : ].reshape(60000, 1)
	return train_image_bin, train_label

@nb.jit
def Estep(train_bin, MU, PI, Z):
    for img_idx in range(60000):
        num_sum = np.full(10, 1, dtype = np.float64)
        for num_idx in range(10):
            for pixel_idx in range(784):
                if train_bin[img_idx][pixel_idx] == 1:
                    num_sum[num_idx] *= MU[num_idx][pixel_idx]
                else:
                    num_sum[num_idx] *= (1 - MU[num_idx][pixel_idx])
            num_sum[num_idx] *= PI[num_idx][0]
        marginal_num = np.sum(num_sum)
        if marginal_num == 0:
            marginal_num = 1
        
        for num_idx in range(10):
            Z[img_idx][num_idx] = num_sum[num_idx] / marginal_num
    return Z

def Mstep(train_bin, MU, PI, Z):
    N_cluster = np.sum(Z, axis=0)
    for num_idx in range(10):
        for pixel_idx in range(784):
            sum_pixel = np.dot(train_bin[:, pixel_idx], Z[:, num_idx])
            marginal = N_cluster[num_idx]
            if marginal == 0:
                marginal = 1
            MU[num_idx][pixel_idx] = sum_pixel / marginal

        PI[num_idx][0] = N_cluster[num_idx] / 60000
        if PI[num_idx][0] == 0:
            PI[num_idx][0] = 1

    return MU, PI

def print_imagination(MU):
    MU_new = MU.copy()
    for num_idx in range(10):
        print('class {}: '.format(num_idx))
        for pixel_idx in range(784):
            if pixel_idx % 28 == 0 and pixel_idx != 0:
                print('')
            if MU_new[num_idx][pixel_idx] >= 0.5:
                print('1', end = ' ')
            else:
                print('0', end = ' ')
        print('\n')

@nb.jit
def label_cluster(train_bin, train_label, MU, PI):
    table = np.zeros(shape = (10, 10), dtype = np.int)
    for img_idx in range(60000):
        num_sum = np.full((10), 1, dtype = np.float64)
        for num_idx in range(10):
            for pixel_idx in range(784):
                if train_bin[img_idx][pixel_idx] == 1:
                    num_sum[num_idx] *= MU[num_idx][pixel_idx]
                else:
                    num_sum[num_idx] *= (1 - MU[num_idx][pixel_idx])
            num_sum[num_idx] *= PI[num_idx][0]
        table[train_label[img_idx][0]][np.argmax(num_sum)] += 1

    '''
    i: real number
    j: predict number
    '''
    # print(' \t[0]\t[1]\t[2]\t[3]\t[4]\t[5]\t[6]\t[7]\t[8]\t[9]\t')
    # for i in range(10):
    #     for j in range(10):
    #         if j == 0:
    #             print('[real {}]'.format(i), end = '')
    #         print('{}\t'.format(table[i][j]), end = '')
    #     print('')

    relation = np.full((10), -1, dtype = np.int)
    for num_idx in range(10):
        '''
        select the biggest value
        than delete the real number value and predict number value
        (change the value into -1)
        '''
        ind = np.unravel_index(np.argmax(table, axis = None), table.shape)
        relation[ind[0]] = ind[1]
        for i in range(10):
            table[i][ind[1]] = -1
            table[ind[0]][i] = -1

    return relation

def print_label(relation, MU):
    for num_idx in range(10):
        cluster = relation[num_idx]
        print('\nlabeled class {}:'.format(num_idx))
        for pixel_idx in range(784):
            if pixel_idx % 28 == 0 and pixel_idx != 0:
                print('')
            if MU[cluster][pixel_idx] >= 0.5:
                print('1', end = ' ')
            else:
                print('0', end = ' ')
        print('')

@nb.jit
def print_confusion_matrix(train_bin, train_label, MU, PI, relation):
    '''
    confusion matrix: (TP, FP, TN, FN)
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    '''
    error = 60000
    confusion_matrix = np.zeros(shape = (10, 4), dtype = np.int)
    for img_idx in range(60000):
        num_sum = np.full(10, 1, dtype = np.float64)
        for num_idx in range(10):
            for pixel_idx in range(784):
                if train_bin[img_idx][pixel_idx] == 1:
                    num_sum[num_idx] *= MU[num_idx][pixel_idx]
                else:
                    num_sum[num_idx] *= (1 - MU[num_idx][pixel_idx])
            num_sum[num_idx] *= PI[num_idx][0]

        predict_cluster = np.argmax(num_sum)
        predict_label = np.where(relation == predict_cluster)

        for num_idx in range(10):
            if num_idx == train_label[img_idx][0]:
                if num_idx == predict_label[0]:
                    confusion_matrix[num_idx][0] += 1
                    error -= 1
                else:
                    confusion_matrix[num_idx][3] += 1
            else:
                if num_idx == predict_label[0]:
                    confusion_matrix[num_idx][1] += 1
                else:
                    confusion_matrix[num_idx][2] += 1

    for num_idx in range(10):
        print('Confusion Matrix {}:'.format(num_idx))
        print('\t\tPredict number {}\tPredict not number {}'.format(num_idx, num_idx))
        print('Is number {}\t\t{}\t\t\t{}'.format(num_idx, confusion_matrix[num_idx][0], confusion_matrix[num_idx][3]))
        print("Isn't number {}\t\t{}\t\t\t{}".format(num_idx, confusion_matrix[num_idx][1], confusion_matrix[num_idx][2]))
        print('\n')
        print('Sensitivity (Successfully predict number {})    : {}'.format(num_idx, confusion_matrix[num_idx][0] / (confusion_matrix[num_idx][0] + confusion_matrix[num_idx][3])))
        print('Specificity (Successfully predict not number {}): {}'.format(num_idx, confusion_matrix[num_idx][2] / (confusion_matrix[num_idx][2] + confusion_matrix[num_idx][1])))
        if num_idx!=9:
            print('---------------------------------------------------------\n')

    return error

'''
number count: 10
pixel count: 28*28 = 784
image count: 60000
'''

if __name__ == '__main__':
    train_bin, train_label = open_file()
    epochs = 20
    total_iter = epochs

    PI = np.random.random_sample((10, 1))
    MU = np.random.random_sample((10, 784))
    MU_prev = np.zeros((10, 784), dtype = np.float64)
    Z  = np.random.random_sample((60000, 10)) 

    for step in range(epochs):
        Z = Estep(train_bin, MU, PI, Z)
        MU, PI = Mstep(train_bin, MU, PI, Z)
        delta = sum(sum(abs(MU - MU_prev)))

        print_imagination(MU)
        print('No. of Iteration: {}, Difference: {}\n'.format(step+1, delta))
        print('---------------------------------------------------------------\n')

        if delta < 10:
            total_iter = step+1
            break

        MU_prev = MU.copy()

    relation = label_cluster(train_bin, train_label, MU, PI)
    print_label(relation, MU)
    print('---------------------------------------------------------------\n')
    error = print_confusion_matrix(train_bin, train_label, MU, PI, relation)
    print('Total iteration to converge: {}'.format(total_iter))
    print('Total error rate: {}'.format(error / 60000))
