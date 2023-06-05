import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size=114, hidden_size=256, num_layers=1):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)

        x = self.relu(hn)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        output = self.sigmoid(x)
        return output


def create_dataset(path_to_dataset: str = "../data/datasets_to_model/5_1_1_wonans_dataset.csv"):
    data = pd.read_csv(path_to_dataset, index_col=0)

    numeric_features = data.dtypes[(data.dtypes == np.float64) |
                                   (data.dtypes == np.int64)].index.tolist()

    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data[numeric_features].drop(["target", "class_target"], axis=1)))

    first_features = scaled_data.columns[0:114]
    second_features = scaled_data.columns[114:228]
    third_features = scaled_data.columns[228:342]
    fourth_features = scaled_data.columns[342:456]
    fifth_features = scaled_data.columns[456:570]

    X = []
    y = data["class_target"].tolist()

    for index, row in tqdm(scaled_data.iterrows(), total=scaled_data.shape[0]):
        X.append([row[first_features].tolist(), row[second_features].tolist(),
                  row[third_features].tolist(), row[fourth_features].tolist(),
                  row[fifth_features].tolist()])

    return torch.Tensor(X), torch.Tensor(y).reshape(len(y), 1)


def train(model: LSTM, X_train: torch.Tensor, y_train: torch.Tensor, n_epochs: int = 1000, learning_rate: float = 0.001,
          X_eval: torch.Tensor = None, y_eval: torch.Tensor = None, eval_per_epochs: int = 10):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model.forward(X_train)

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        eval_losses = []
        accuracies = []

        if epoch % eval_per_epochs == 0:
            logits = model.forward(X_eval)
            eval_loss = criterion(logits, y_eval)
            preds = (logits > 0.5).detach().numpy()

            accuracy = accuracy_score(y_eval.flatten(), preds)
            f1 = f1_score(y_eval.flatten(), preds)
            recall = recall_score(y_eval.flatten(), preds)
            precision = precision_score(y_eval.flatten(), preds)

            eval_losses.append(eval_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {epoch}, train_loss: {loss.item()}, eval_loss: {eval_loss}, "
                  f"accuracy: {accuracy}, f1: {f1}, recall: {recall}, precision: {precision}")

    return model


X, y = create_dataset()

X_train, y_train = X[3000:], y[3000:]
X_eval, y_eval = X[:3000], y[:3000]

print(np.unique(y_eval.numpy().flatten(), return_counts=True))

model = LSTM()
model, losses, accs = train(model, X_train, y_train,
                            n_epochs=1000,
                            learning_rate=0.1,
                            eval_per_epochs=1,
                            X_eval=X_eval,
                            y_eval=y_eval)

plt.plot(losses)
plt.plot(accs)
