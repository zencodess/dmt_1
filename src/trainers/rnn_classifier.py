import torch
import torch.nn as torch_nn

from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class RNNClassifier(torch_nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.rnn = torch_nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dense = torch_nn.Linear(hidden_dim, 1)
        self.sigmoid = torch_nn.Sigmoid()
        self.model = None

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # use last time step
        out = self.dense(out)
        return self.sigmoid(out).squeeze()

    def train_rnn_model(self, train_x, train_y, val_x, val_y, device='cpu', epochs=20, batch_size=32, lr=0.001):
        model = RNNClassifier(self.input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch_nn.BCELoss()

        train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y).float())
        val_dataset = TensorDataset(torch.tensor(val_x), torch.tensor(val_y).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    pred = model(xb).cpu().numpy()
                    all_preds.extend(pred)
                    all_labels.extend(yb.numpy())

            all_preds = [1 if p >= 0.5 else 0 for p in all_preds]
            f1 = f1_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_preds)
            print(f"Epoch {epoch + 1}: F1 = {f1:.4f}, AUC = {auc:.4f}")
        self.model = model

    def save_model(self):
        torch.save(self.model.state_dict(), "models/rnn_classifier.pth")
        print("RNN model saved to models/rnn_classifier.pth")

    def test_rnn_model(self, test_x, test_y, device='cpu', epochs=20):
        pass