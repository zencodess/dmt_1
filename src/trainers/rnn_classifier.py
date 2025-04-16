import torch
import torch.nn as torch_nn
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class RNNClassifier(torch_nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.rnn = torch_nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dense = torch_nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dense(out[:, -1, :])
        return out.squeeze()

    def train_rnn_model(self, train_x, train_y, val_x, val_y, device='cpu', epochs=500, batch_size=32, lr=0.0001):
        self.to(device)
        print("Train label balance:", np.bincount(train_y))
        print("Val label balance:", np.bincount(val_y))
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = torch_nn.BCEWithLogitsLoss()

        train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y).float())
        val_dataset = TensorDataset(torch.tensor(val_x), torch.tensor(val_y).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = self(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            self.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    pred = self(xb)
                    prob = torch.sigmoid(pred).cpu().numpy()
                    preds = np.round(prob)
                    all_preds.extend(preds)
                    all_labels.extend(yb.numpy())

            f1 = f1_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_preds)
            print(f"Epoch {epoch + 1}: F1 = {f1:.4f}, AUC = {auc:.4f}")
        self.save_model()

    def save_model(self):
        torch.save(self.state_dict(), "models/rnn_classifier.pth")
        print("RNN model saved to models/rnn_classifier.pth")

    def test_rnn_model(self, test_x, test_y, device='cpu'):
        self.eval()
        self.to(device)

        test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y).float())
        test_loader = DataLoader(test_dataset, batch_size=32)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred = self(xb)
                prob = torch.sigmoid(pred).cpu().numpy()
                preds = np.round(prob)
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Test Results — F1: {f1:.4f}, AUC: {auc:.4f}")