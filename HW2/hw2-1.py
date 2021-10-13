import argparse
import numpy as np
np.set_printoptions(precision=3, suppress=True)
'''
python hw2-1.py --TRAINING_IMAGE=train-images-idx3-ubyte --TRAINING_LABEL=train-labels-idx1-ubyte --TESTING_IMAGE=t10k-images-idx3-ubyte --TESTING_LABEL=t10k-labels-idx1-ubyte --OPTION=0
'''

def open_train_file(training_image_path,training_label_path):
    file_train_image = open(training_image_path, 'rb')
    file_train_label = open(training_label_path, 'rb')

    global train_image_magic, train_image_number, train_image_row, train_image_col, train_label_magic, train_label_total_count
    train_image_magic = int.from_bytes(file_train_image.read(4), byteorder = 'big')
    train_image_number = int.from_bytes(file_train_image.read(4), byteorder = 'big')
    train_image_row = int.from_bytes(file_train_image.read(4), byteorder = 'big')
    train_image_col = int.from_bytes(file_train_image.read(4), byteorder = 'big')
    train_label_magic = int.from_bytes(file_train_label.read(4), byteorder = 'big')
    train_label_total_count = int.from_bytes(file_train_label.read(4), byteorder = 'big')

    return file_train_image, file_train_label

def open_test_file(testing_image_path,testing_label_path):
    file_test_image = open(testing_image_path, 'rb')
    file_test_label = open(testing_label_path, 'rb')

    global test_image_magic, test_image_number, test_image_row, test_image_col, test_label_magic, test_label_total_count
    test_image_magic = int.from_bytes(file_test_image.read(4), byteorder = 'big')
    test_image_number = int.from_bytes(file_test_image.read(4), byteorder = 'big')
    test_image_row = int.from_bytes(file_test_image.read(4), byteorder = 'big')
    test_image_col = int.from_bytes(file_test_image.read(4), byteorder = 'big')
    test_label_magic = int.from_bytes(file_test_label.read(4), byteorder = 'big')
    test_label_total_count = int.from_bytes(file_test_label.read(4), byteorder = 'big')

    return file_test_image, file_test_label

def normalization(prob):
    temp = 0
    for j in range(10):
        temp += prob[j]
    for j in range(10):
        prob[j] /= temp
    return prob

def print_result(prob, ans):
    print('Posterior (in log scale):')
    for j in range(10):
        print(j, ':', prob[j])
    pred = np.argmin(prob)
    print('Prediction: ', pred, ', Ans: ', ans, '\n')

    if pred == ans:
        return 0
    else:
        return 1

def print_imagination_discrete(likelihood):
    print('Imagination of numbers in Bayesian classifier:', '\n')
    for i in range(10):
        print(i, ':')
        for j in range(28):
            for k in range(28):
                temp = 0
                for t in range(16): # bin 0~15
                    temp += likelihood[i][j * 28 + k][t]
                for t in range(16, 32): # bin 16~32
                    temp -= likelihood[i][j * 28 + k][t]
                if temp > 0:
                    print('0', end = ' ')
                else:
                    print('1', end = ' ')
            print('\n')
        print('\n')

def discrete_mode(training_image_path, training_label_path, testing_image_path, testing_label_path):
    ## Training   
    file_train_image, file_train_label = open_train_file(training_image_path, training_label_path)
    prior = np.zeros((10), dtype = int)
    likelihood = np.zeros((10, train_image_row * train_image_col, 32), dtype = int)

    for i in range(train_image_number):
        label = int.from_bytes(file_train_label.read(1), byteorder = 'big')
        prior[label] += 1
        for pixel_idx in range(train_image_row * train_image_col):
            pixel_value = int.from_bytes(file_train_image.read(1), byteorder = 'big')
            likelihood[label][pixel_idx][int(pixel_value/8)] += 1

    likelihood_sum = np.zeros((10, train_image_row * train_image_col), dtype = int)
    for i in range(10):
        for j in range(train_image_row * train_image_col):
            for k in range(32):
                likelihood_sum[i][j] += likelihood[i][j][k]

    # print('prior: ',prior)
    # print('likelihood: ',likelihood)
    # print('likelihood_sum: ',likelihood_sum)

    file_train_image.close()
    file_train_label.close()

    ## Testing
    file_test_image, file_test_label = open_test_file(testing_image_path, testing_label_path)
    error = 0
    for i in range(test_image_number):
        # print('index: {}'.format(i))
        answer = int.from_bytes(file_test_label.read(1), byteorder = 'big')
        probability = np.zeros((10), dtype = float)
        test_image = np.zeros((test_image_row * test_image_col), dtype = int)
        for j in range(test_image_row * test_image_col):
            test_image[j] = int((int.from_bytes(file_test_image.read(1), byteorder = 'big')) / 8)
        for j in range(10):
            probability[j] += np.log(float(prior[j] / train_image_number))
            for k in range(test_image_row * test_image_col):
                temp = likelihood[j][k][test_image[k]]
                if temp == 0:
                    probability[j] += np.log(float(1e-6 / likelihood_sum[j][k]))
                else:
                    probability[j] += np.log(float(likelihood[j][k][test_image[k]] / likelihood_sum[j][k]))
        # print('probability = {}'.format(probability))
        probability = normalization(probability)
        error += print_result(probability, answer)
    
    file_test_image.close()
    file_test_label.close()

    print_imagination_discrete(likelihood)
    print('Error rate: ', float(error / test_image_number))

def Gaussian_distribution(value, mean, var):
    return np.log(1.0 / (np.sqrt(2.0 * np.pi * var))) - ((value - mean)**2.0 / (2.0 * var))

def print_imagination_continuous(likelihood):
    print('Imagination of numbers in Bayesian classifier:', '\n')
    for i in range(10):
        print(i, ':')
        for j in range(28):
            for k in range(28):
                if likelihood[i][j * 28 + k] < 128:
                    print('0', end = ' ')
                else:
                    print('1', end = ' ')
            print('\n')
        print('\n')

def continuous_mode(training_image_path, training_label_path, testing_image_path, testing_label_path):
    ## Training   
    file_train_image, file_train_label = open_train_file(training_image_path, training_label_path)
    prior = np.zeros((10), dtype = float)
    pixel_square = np.zeros((10, train_image_row * train_image_col), dtype = float)
    pixel_mean = np.zeros((10, train_image_row * train_image_col), dtype = float)
    pixel_var = np.zeros((10, train_image_row * train_image_col), dtype = float)

    for i in range(train_image_number):
        label = int.from_bytes(file_train_label.read(1), byteorder = 'big')
        prior[label] += 1
        for pixel_idx in range(train_image_row * train_image_col):
            pixel_value = int.from_bytes(file_train_image.read(1), byteorder = 'big')
            pixel_square[label][pixel_idx] += (pixel_value ** 2)
            pixel_mean[label][pixel_idx] += pixel_value

    #Calculate mean and standard deviation
    for label in range(10):
        for pixel_idx in range(train_image_row * train_image_col):
            pixel_mean[label][pixel_idx] = float(pixel_mean[label][pixel_idx] / prior[label])
            pixel_var[label][pixel_idx] = float(pixel_square[label][pixel_idx] / prior[label]) - float(pixel_mean[label][pixel_idx] ** 2)
    
            # psuedo count for variance
            if pixel_var[label][pixel_idx] == 0:
                pixel_var[label][pixel_idx] = 1e-4
    prior = prior / train_image_number
    prior = np.log(prior)
    
    ## Testing
    file_test_image, file_test_label = open_test_file(testing_image_path, testing_label_path)

    error = 0
    for i in range(10000):
        # print('index: {}'.format(i))
        answer = int.from_bytes(file_test_label.read(1), byteorder = 'big')
        probability = np.zeros((10), dtype = float)
        test_image = np.zeros((28 * 28), dtype = float)
        for pixel_idx in range(28 * 28):
            test_image[pixel_idx] = int.from_bytes(file_test_image.read(1), byteorder = 'big')
        for label in range(10):
            probability[label] += prior[label]
            for pixel_idx in range(test_image_row * test_image_col):
                testing_value = Gaussian_distribution(test_image[pixel_idx], pixel_mean[label][pixel_idx], pixel_var[label][pixel_idx])
                probability[label] += testing_value

        # print('probability = {}'.format(probability))
        probability = normalization(probability)
        error += print_result(probability, answer)
    print_imagination_continuous(pixel_mean)
    print('Error rate: ', float(error / test_image_number))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAINING_IMAGE', type = str, default = 'train-images-idx3-ubyte')
    parser.add_argument('--TRAINING_LABEL', type = str, default = 'train-labels-idx1-ubyte')
    parser.add_argument('--TESTING_IMAGE', type = str, default = 't10k-images-idx3-ubyte')
    parser.add_argument('--TESTING_LABEL', type = str, default = 't10k-labels-idx1-ubyte')
    parser.add_argument('--OPTION', type = int, default = 0)
    args = parser.parse_args()

    training_image_path = args.TRAINING_IMAGE
    training_label_path = args.TRAINING_LABEL
    testing_image_path = args.TESTING_IMAGE
    testing_label_path = args.TESTING_LABEL
    toggle_mode = args.OPTION

    if toggle_mode == 0:
        discrete_mode(training_image_path, training_label_path, testing_image_path, testing_label_path)
    else:
        continuous_mode(training_image_path, training_label_path, testing_image_path, testing_label_path)