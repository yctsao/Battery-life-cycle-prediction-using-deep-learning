import torch
import pandas as pd
import torch.nn as nn
import random
import torch.optim as optim
from sklearn import preprocessing

n_cyc = 5 # using first n_cyc to calculate battery life

X_train_path = './data/X_train_first ' + str(n_cyc) + ' cyc_60%.csv'
Y_train_path = './data/Y_train_first ' + str(n_cyc) + ' cyc_60%.csv'
X_dev_path = './data/X_dev_first ' + str(n_cyc) + ' cyc_20%.csv'
Y_dev_path = './data/Y_dev_first ' + str(n_cyc) + ' cyc_20%.csv'
X_test_path = './data/X_test_first ' + str(n_cyc) + ' cyc_20%.csv'
Y_test_path = './data/Y_test_first ' + str(n_cyc) + ' cyc_20%.csv'

# define the demision of the parameters of RNN
n_hidden = 100
n_features = 8
n_categories = 1

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.tanh(hidden)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        output = output.reshape(1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def loadata(X_train_path, Y_train_path, X_dev_path, Y_dev_path, X_test_path, Y_test_path):
    X_train = torch.tensor((pd.read_csv(X_train_path, header= None).T).values).float()
    Y_train = pd.read_csv(Y_train_path, header= None).T
    Y_train_class = torch.tensor((Y_train >= 550).values).float()

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
    return X_train, Y_train, Y_train_class, X_dev, Y_dev, Y_dev_class, X_test, Y_test, Y_test_class

def reshapeData(dataset):
    m, n = dataset.shape
    n_features = 8
    T_x = int(dataset.shape[1] / n_features)
    new_dateset = torch.zeros((m, n))
    for i in range(T_x):
        features_per_cyc = dataset[:, list(range(i, n, T_x))]
        new_dateset[:, i*n_features:(i+1)*n_features] = features_per_cyc
    new_dateset = torch.reshape(new_dateset, (m, T_x, 1, n_features)) # T_x = # of cycles, n_features = 8,
    return new_dateset

def randomTrainingExample(X, Y):
    m = X.shape[0]
    random_index = random.randint(0, m-1)
    selected_battery = X[random_index]
    life_selected_battery = Y[random_index]
    return selected_battery, life_selected_battery

def train(x, y):
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    for i in range(x.size()[0]):
        output, hidden = rnn(x[i], hidden)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return output, loss

# output the prediction for 1 training example
def evaluate(x, y):
    hidden = rnn.initHidden()
    for i in range(x.size()[0]):
        output, hidden = rnn(x[i], hidden)
    loss = criterion(output, y)
    return output, loss

# Define evaluation matric
def calculate_accuracy(X, Y):
    m = X.shape[0]
    outputs = torch.zeros((m, 1))
    for i in range(m):
        output, loss = evaluate(X[i], Y[i])
        outputs[i,:] = output
    Y_pred = outputs >= 0.5
    Y_pred = Y_pred.float()
    num_correct = torch.sum(Y == Y_pred)
    acc = (num_correct * 100.0 / len(Y)).item()
    return acc

X_train, Y_train, Y_train_class, X_dev, Y_dev, Y_dev_class, X_test, Y_test, Y_test_class = loadata(X_train_path, Y_train_path, X_dev_path, Y_dev_path, X_test_path, Y_test_path)

# normalizing the data
X_train_norm = torch.tensor(preprocessing.scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)).float()
X_dev_norm = torch.tensor(preprocessing.scale(X_dev, axis=0, with_mean=True, with_std=True, copy=True)).float()
X_test_norm = torch.tensor(preprocessing.scale(X_test, axis=0, with_mean=True, with_std=True, copy=True)).float()

# reshape the data
X_train_norm_reshaped = reshapeData(X_train_norm)
X_dev_norm_reshaped = reshapeData(X_dev_norm)
X_test_norm_reshaped = reshapeData(X_test_norm)

n_iters = 30000
print_every = 1000
plot_every = 100

# Keep track of losses for plotting
all_losses_diffLR = {}

learning_rate = [0.5, 0.05, 0.005, 0.0005, 0.00005]

for i in range(len(learning_rate)):
    train_current_loss = 0
    train_all_losses = []

    rnn = RNN(n_features, n_hidden, n_categories)
    # Define the cost function
    criterion = nn.BCELoss()
    # Define the optimizer
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate[i])

    for iter in range(1, n_iters + 1):
        train_selected_battery, train_battery_life = randomTrainingExample(X_train_norm_reshaped, Y_train_class)
        train_output, train_loss = train(train_selected_battery, train_battery_life)
        train_loss = train_loss.item()
        train_current_loss += train_loss  # Add current loss avg to list of losses

        if iter % plot_every == 0:
            train_all_losses.append(train_current_loss / plot_every)
            train_current_loss = 0

    all_losses_diffLR[learning_rate[i]] = train_all_losses

    training_accuracy = calculate_accuracy(X_train_norm_reshaped, Y_train_class)
    dev_accuracy = calculate_accuracy(X_dev_norm_reshaped, Y_dev_class)
    test_accuracy = calculate_accuracy(X_test_norm_reshaped, Y_test_class)
    print('training accuracy with lr =  ' + str(learning_rate[i]) + ': ' + str(training_accuracy))
    print('dev accuracy with lr = ' + str(learning_rate[i]) + ': ' + str(dev_accuracy))
    print('test accuracy with lr = ' + str(learning_rate[i]) + ': ' + str(test_accuracy))





