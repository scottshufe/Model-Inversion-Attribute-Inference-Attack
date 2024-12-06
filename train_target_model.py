from utils_maia import load_data_adult

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, n_inputs) -> None:
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(8, 2)
        )

    def forward(self, data, target=None):
        out = self.model(data)

        return out


def compute_metrics(yout, target=None):
    if target is None:
        return {}
    # yout = yout.detach().cpu().numpy() > 0.50
    yout = torch.argmax(yout, dim=1).detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    return {
        "accuracy_score": metrics.accuracy_score(target, yout),
        "f1": metrics.f1_score(target, yout),
        "precision": metrics.precision_score(target, yout),
        "reall": metrics.recall_score(target, yout)
    }


if __name__ == '__main__':
    _, df, _, _ = load_data_adult()

    train_data = df.values[:, :-1]
    train_targets = df.values[:, -1]
    train_x_tensor = torch.Tensor(train_data)
    train_y_tensor = torch.LongTensor(train_targets)
    train_y_tensor_onehot = F.one_hot(train_y_tensor).float()

    epochs = 1000
    model = Model(n_inputs=train_data.shape[1])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # scores = {}
        model.train()

        optimizer.zero_grad()
        yout = model(train_x_tensor, train_y_tensor_onehot)
        loss_fn = criterion(yout, train_y_tensor_onehot)
        scores = compute_metrics(yout, train_y_tensor)
        loss_fn.backward()
        optimizer.step()

        print(
            "[Train]: epoch: {0}, loss: {1}, score: {2}".format(epoch, loss_fn, scores)
        )

    PATH = "tmp_model/target_model_adult_dnn.pt"
    torch.save(model.state_dict(), PATH)
