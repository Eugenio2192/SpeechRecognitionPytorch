import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
DATA_PATH = "data.json"

LEARNING_RATE = 0.001
EPOCHS = 40
BATCH_SIZE = 32

SAVED_MODEL_PATH = "MODEL/model.ckpt"

NUM_KEYWORDS = 8

def load_dataset(data_path):

    with open(data_path, "r") as fp:
        data= json.load(fp)

        # Extract inputs and targets
        X = np.array(data["MFCCs"])
        y = np.array(data["labels"])

        return X, y


def get_data_splits(data_path, test_size = 0.1, test_validation = 0.1):

    # load data path
    X, y = load_dataset(data_path)
    # create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)
    # convert inputs from 2d to 3d
    y_train = Variable(torch.Tensor(y_train).long())
    y_test = Variable(torch.Tensor(y_test).long())
    # y_validation = Variable(torch.Tensor(y_validation).long())
    X_train = Variable(torch.Tensor(X_train[...,np.newaxis]).float()).transpose(1,3)
    # X_validation = Variable(torch.Tensor(X_validation[...,np.newaxis]).float()).transpose(1,3)
    X_test = Variable(torch.tensor(X_test[...,np.newaxis]).float()).transpose(1,3)
    return X_train, X_test, y_train, y_test


class SpeechRecognitionNet(torch.nn.Module):
    def __init__(self):
        super(SpeechRecognitionNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=(1,1))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=(1,1))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,1))
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160,64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, NUM_KEYWORDS)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def main():

    # load train/validation/test data splits
    X_train, X_test, y_train, y_test = get_data_splits(DATA_PATH)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    # validation_loader = DataLoader(TensorDataset(X_validation, y_validation), batch_size=BATCH_SIZE, shuffle=False)

    # build the CNN
    model = SpeechRecognitionNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses, accuracies = [], []
    total_step = len(train_loader)
    model.train()
    for epoch in range(EPOCHS):

        for i, (images, labels) in enumerate(t := tqdm(train_loader)):
            # Run forward
            outputs = model(images)
            loss = loss_function(outputs, labels)
            losses.append(loss.item())

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track Accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct/total
            accuracies.append(accuracy)
            # print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}")
            t.set_description("Epoch: %.f Loss %.2f, accuracy %.2f" % (epoch, loss, accuracy))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"The test accuracy of the model on the test audio files is: {(correct/ total)*100} %")

    torch.save(model.state_dict(), SAVED_MODEL_PATH)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(range(len(accuracies)), accuracies)
    ax.plot(range(len(losses)), losses)
    plt.show()


if __name__ == "__main__":
    main()