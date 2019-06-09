import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from pytorchtools import EarlyStopping


n_cyc = 100
print('battery cycles: ', n_cyc)

X_train_path = './data/X_train_shuffled_first ' + str(n_cyc) + ' cyc_60%.csv'
Y_train_path = './data/Y_train_shuffled_first ' + str(n_cyc) + ' cyc_60%.csv'
X_dev_path = './data/X_dev_shuffled_first ' + str(n_cyc) + ' cyc_20%.csv'
Y_dev_path = './data/Y_dev_shuffled_first ' + str(n_cyc) + ' cyc_20%.csv'
X_test_path = './data/X_test_shuffled_first ' + str(n_cyc) + ' cyc_20%.csv'
Y_test_path = './data/Y_test_shuffled_first ' + str(n_cyc) + ' cyc_20%.csv'

# read the csv file
X_train = pd.read_csv(X_train_path, header= None).T
Y_train = pd.read_csv(Y_train_path, header= None).T
Y_train_class = (Y_train >= 550)
X_train = torch.tensor(X_train.values).float()
Y_train_class = torch.tensor(Y_train_class.values).float()

X_dev = pd.read_csv(X_dev_path, header= None).T
Y_dev = pd.read_csv(Y_dev_path, header= None).T
Y_dev_class = (Y_dev >= 550)
X_dev = torch.tensor(X_dev.values).float()
Y_dev_class = torch.tensor(Y_dev_class.values).float()

X_test = pd.read_csv(X_test_path, header= None).T
Y_test = pd.read_csv(Y_test_path, header= None).T
Y_test_class = (Y_test >= 550)
X_test = torch.tensor(X_test.values).float()
Y_test_class = torch.tensor(Y_test_class.values).float()

# data normalization
X_train_norm = preprocessing.scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
X_train_norm = torch.tensor(X_train_norm).float()
X_dev_norm = preprocessing.scale(X_dev, axis=0, with_mean=True, with_std=True, copy=True)
X_dev_norm = torch.tensor(X_dev_norm).float()
X_test_norm = preprocessing.scale(X_test, axis=0, with_mean=True, with_std=True, copy=True)
X_test_norm = torch.tensor(X_test_norm).float()

# Define evaluation matric
def calculate_accuracy(X, Y):
    outputs = model(X)
    Y_pred = outputs >= 0.5
    Y_pred = Y_pred.float()
    num_correct = torch.sum(Y == Y_pred)
    acc = (num_correct * 100.0 / len(Y)).item()
    return acc

learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]

train_acc = {}
dev_acc = {}
test_acc = {}


# early stopping patience; how long to wait after last time validation loss improved.
patience = 20

for j in range(len(learning_rate)):
    model = nn.Sequential(
        nn.Linear(8*n_cyc, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        nn.Sigmoid())

    # Define the cost function
    criterion = nn.BCELoss()
    # Define the optimizer, learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate[j])
    # Record cost per interation
    cost_per_iteration = []
    dev_cost_per_iteration = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    if learning_rate[j] not in (train_acc.keys() and dev_acc.keys() and test_acc.keys()) :
        train_acc[learning_rate[j]] = []
        dev_acc[learning_rate[j]] = []
        test_acc[learning_rate[j]] = []

    # Train the network on the training data
    for i in range(10000):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward propogation
        outputs = model(X_train_norm)
        outputs_dev = model(X_dev_norm)
        # calculate the loss
        loss = criterion(outputs, Y_train_class)
        loss_dev = criterion(outputs_dev, Y_dev_class)
        # backpropogation + update parameters
        loss.backward()
        optimizer.step()

        cost = loss.item()
        cost_per_iteration.append(cost)

        cost_dev = loss_dev.item()
        dev_cost_per_iteration.append(cost_dev)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(loss_dev, model)

        if early_stopping.early_stop:
            print("Early stopping at " + str(i) + 'th iterations')
            break

    # Calculate accuracy
    training_accuracy = calculate_accuracy(X_train_norm, Y_train_class)
    dev_accuracy = calculate_accuracy(X_dev_norm, Y_dev_class)
    test_accuracy = calculate_accuracy(X_test_norm, Y_test_class)
    print('training accuracy with lr =  ' + str(learning_rate[j]) + ': ' + str(training_accuracy))
    print('dev accuracy with lr = ' + str(learning_rate[j]) + ': ' + str(dev_accuracy))
    print('test accuracy with lr = ' + str(learning_rate[j]) + ': ' + str(test_accuracy))

