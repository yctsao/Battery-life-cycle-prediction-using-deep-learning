import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle

matFilename_1 = './data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
matFilename_2 = './data/2017-06-30_batchdata_updated_struct_errorcorrect.mat'
matFilename_3 = './data/2018-04-12_batchdata_updated_struct_errorcorrect.mat'

f_1 = h5py.File(matFilename_1)
f_2 = h5py.File(matFilename_2)
f_3 = h5py.File(matFilename_3)

batch_1 = f_1['batch']
batch_2 = f_2['batch']
batch_3 = f_3['batch']

num_cells_1 = batch_1['summary'].shape[0]
num_cells_2 = batch_2['summary'].shape[0]
num_cells_3 = batch_3['summary'].shape[0]

n_cyc = 25   # only use first 100 cycles data to train model

for i in range(num_cells_1):
    if i != 8 and i != 10 and i != 12 and i != 13 and i != 22:
        cl = f_1[batch_1['cycle_life'][i, 0]].value
        # policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
        summary_cur_cell = np.vstack(f_1[batch_1['summary'][i, 0]]['IR'][0, 0:n_cyc].reshape(n_cyc, 1))
        summary_cur_cell = np.vstack((summary_cur_cell, f_1[batch_1['summary'][i, 0]]['QCharge'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_1[batch_1['summary'][i, 0]]['QDischarge'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_1[batch_1['summary'][i, 0]]['Tavg'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_1[batch_1['summary'][i, 0]]['Tmin'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_1[batch_1['summary'][i, 0]]['Tmax'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_1[batch_1['summary'][i, 0]]['chargetime'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_1[batch_1['summary'][i, 0]]['cycle'][0, 0:n_cyc].reshape(n_cyc, 1)))

        if i == 0:
            summary_X = np.hstack(summary_cur_cell).reshape(summary_cur_cell.shape[0], 1)
            summary_Y = np.hstack(cl).reshape(1, 1)

        else:
            summary_X = np.hstack((summary_X, summary_cur_cell))
            summary_Y = np.hstack((summary_Y, cl))
print('after 1st batch: ', summary_X.shape)

for i in range(num_cells_2):
    if i != 7 and i != 8 and i != 9 and i != 15 and i != 16:
        cl = f_2[batch_2['cycle_life'][i, 0]].value
        # policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
        summary_cur_cell = np.vstack(f_2[batch_2['summary'][i, 0]]['IR'][0, 0:n_cyc].reshape(n_cyc, 1))
        summary_cur_cell = np.vstack((summary_cur_cell, f_2[batch_2['summary'][i, 0]]['QCharge'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_2[batch_2['summary'][i, 0]]['QDischarge'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_2[batch_2['summary'][i, 0]]['Tavg'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_2[batch_2['summary'][i, 0]]['Tmin'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_2[batch_2['summary'][i, 0]]['Tmax'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_2[batch_2['summary'][i, 0]]['chargetime'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack((summary_cur_cell, f_2[batch_2['summary'][i, 0]]['cycle'][0, 0:n_cyc].reshape(n_cyc, 1)))

        summary_X = np.hstack((summary_X, summary_cur_cell))
        summary_Y = np.hstack((summary_Y, cl))

print('after 1st + 2nd batch: ', summary_X.shape)

for i in range(num_cells_3):
    if i != 2 and i != 23 and i != 32 and i != 37 and i != 38 and i != 39:
        cl = f_3[batch_3['cycle_life'][i, 0]].value
        # policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
        summary_cur_cell = np.vstack(f_3[batch_3['summary'][i, 0]]['IR'][0, 0:n_cyc].reshape(n_cyc, 1))
        summary_cur_cell = np.vstack(
            (summary_cur_cell, f_3[batch_3['summary'][i, 0]]['QCharge'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack(
            (summary_cur_cell, f_3[batch_3['summary'][i, 0]]['QDischarge'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack(
            (summary_cur_cell, f_3[batch_3['summary'][i, 0]]['Tavg'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack(
            (summary_cur_cell, f_3[batch_3['summary'][i, 0]]['Tmin'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack(
            (summary_cur_cell, f_3[batch_3['summary'][i, 0]]['Tmax'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack(
            (summary_cur_cell, f_3[batch_3['summary'][i, 0]]['chargetime'][0, 0:n_cyc].reshape(n_cyc, 1)))
        summary_cur_cell = np.vstack(
            (summary_cur_cell, f_3[batch_3['summary'][i, 0]]['cycle'][0, 0:n_cyc].reshape(n_cyc, 1)))

        summary_X = np.hstack((summary_X, summary_cur_cell))
        summary_Y = np.hstack((summary_Y, cl))

print('after 1st + 2nd + 3rd batch: ', summary_X.shape)

np.savetxt('./data/X_combined.csv', summary_X, delimiter=',')
np.savetxt('./data/Y_combined.csv', summary_Y, delimiter=',')


data_size = summary_X.shape[1]  # total dataset = 124
print('data size: ', data_size)

summary_X = summary_X.T
summary_Y = summary_Y.T
print('summary_X.shape[0]: ', summary_X.shape[0])
# random selection: 10% for test set
random_indexes_test = np.random.choice(np.arange(summary_X.shape[0]), size=int(data_size * 0.2), replace=False)
print('random_indexes_test: ', random_indexes_test)
X_test = summary_X[random_indexes_test]  # ?
Y_test = summary_Y[random_indexes_test]
np.savetxt('./data/X_test_first ' + str(n_cyc) + ' cyc_20%.csv', X_test.T, delimiter=',')
np.savetxt('./data/Y_test_first ' + str(n_cyc) + ' cyc_20%.csv', Y_test.T, delimiter=',')

# remove the selected test set
summary_X = np.delete(summary_X, random_indexes_test, 0)
summary_Y = np.delete(summary_Y, random_indexes_test, 0)
print('X after removing 10% for test set: ', summary_X.shape)
print('Y after removing 10% for test set: ', summary_Y.shape)

# random selection: 10% for dev set
random_indexes_dev = np.random.choice(np.arange(summary_X.shape[0]), size=int(data_size * 0.2), replace=False)
print('random_indexes_dev: ', random_indexes_dev)
X_dev = summary_X[random_indexes_dev]
Y_dev = summary_Y[random_indexes_dev]
np.savetxt('./data/X_dev_first ' + str(n_cyc) + ' cyc_20%.csv', X_dev.T, delimiter=',')
np.savetxt('./data/Y_dev_first ' + str(n_cyc) + ' cyc_20%.csv', Y_dev.T, delimiter=',')

summary_X = np.delete(summary_X, random_indexes_dev, 0)
summary_Y = np.delete(summary_Y, random_indexes_dev, 0)
print('X after removing 10% for dev set: ', summary_X.shape)
print('Y after removing 10% for dev set: ', summary_Y.shape)

np.savetxt('./data/X_train_first ' + str(n_cyc) + ' cyc_60%.csv', summary_X.T, delimiter=',')
np.savetxt('./data/Y_train_first ' + str(n_cyc) + ' cyc_60%.csv', summary_Y.T, delimiter=',')
