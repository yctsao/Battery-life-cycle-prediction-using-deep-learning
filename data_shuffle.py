import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

n_cyc = 10

X_train_path = './data/X_train_first ' + str(n_cyc) + ' cyc_60%.csv'
Y_train_path = './data/Y_train_first ' + str(n_cyc) + ' cyc_60%.csv'
X_dev_path = './data/X_dev_first ' + str(n_cyc) + ' cyc_20%.csv'
Y_dev_path = './data/Y_dev_first ' + str(n_cyc) + ' cyc_20%.csv'
X_test_path = './data/X_test_first ' + str(n_cyc) + ' cyc_20%.csv'
Y_test_path = './data/Y_test_first ' + str(n_cyc) + ' cyc_20%.csv'

X_train = pd.read_csv(X_train_path, header= None).values
Y_train = pd.read_csv(Y_train_path, header= None).values
X_dev = pd.read_csv(X_dev_path, header= None).values
Y_dev = pd.read_csv(Y_dev_path, header= None).values
X_test = pd.read_csv(X_test_path, header= None).values
Y_test = pd.read_csv(Y_test_path, header= None).values

train_size = X_train.shape[1]  # m_train= 100
dev_size = X_dev.shape[1]  # m_dev= 12
test_size = X_test.shape[1]  # m_train= 12

# random shuffling train, dev, test set
random_indexes_train = np.random.choice(train_size, size=train_size, replace=False)
print('random_indexes_train: ', random_indexes_train)
X_train_shuffled = X_train[:, random_indexes_train]
Y_train_shuffled = Y_train[:, random_indexes_train]
np.savetxt('./data/X_train_shuffled_first ' + str(n_cyc) + ' cyc_60%.csv', X_train_shuffled, delimiter=',')
np.savetxt('./data/Y_train_shuffled_first ' + str(n_cyc) + ' cyc_60%.csv', Y_train_shuffled, delimiter=',')

random_indexes_dev = np.random.choice(dev_size, size=dev_size, replace=False)
print('random_indexes_dev: ', random_indexes_dev)
X_dev_shuffled = X_dev[:, random_indexes_dev]
Y_dev_shuffled = Y_dev[:, random_indexes_dev]
np.savetxt('./data/X_dev_shuffled_first ' + str(n_cyc) + ' cyc_20%.csv', X_dev_shuffled, delimiter=',')
np.savetxt('./data/Y_dev_shuffled_first ' + str(n_cyc) + ' cyc_20%.csv', Y_dev_shuffled, delimiter=',')

random_indexes_test = np.random.choice(test_size, size=test_size, replace=False)
print('random_indexes_test: ', random_indexes_test)
X_test_shuffled = X_test[:, random_indexes_test]
Y_test_shuffled = Y_test[:, random_indexes_test]
np.savetxt('./data/X_test_shuffled_first ' + str(n_cyc) + ' cyc_20%.csv', X_test_shuffled, delimiter=',')
np.savetxt('./data/Y_test_shuffled_first ' + str(n_cyc) + ' cyc_20%.csv', Y_test_shuffled, delimiter=',')

