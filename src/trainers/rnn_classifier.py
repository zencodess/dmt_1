import datetime
import torch
import torch.nn as torch_nn
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class RNNClassifier(torch_nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.rnn = torch_nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout) # just change to GRU for GRU experiments
        self.dense = torch_nn.Linear(hidden_dim, 1)

    def forward(self, x): # (batch, seq, features)
        out, _ = self.rnn(x) # (batch, seq, hidden)
        out = self.dense(out[:, -1, :]) # (batch, 1)
        return out.squeeze(), 0

    def train_rnn_model(self, train_x, train_y, val_x, val_y, epochs=500, batch_size=32, lr=0.005, prob_treshold=0.52):
        # print("train label balance:", np.bincount(train_y))
        # print("val label balance:", np.bincount(val_y))

        optimizer = optim.Adam(self.parameters(), lr=lr)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, alpha=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = torch_nn.BCEWithLogitsLoss()

        train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y).float())
        val_dataset = TensorDataset(torch.tensor(val_x), torch.tensor(val_y).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                pred, _ = self(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            self.eval()
            all_preds, all_labels = [], []
            val_probs = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred, _ = self(xb)
                    prob = torch.sigmoid(pred).numpy()
                    val_probs.extend(prob)
                    # preds = np.round(prob)
                    # all_preds.extend(preds)
                    all_labels.extend(yb.numpy())

            # prob threshold experiments to maximize F1
            # best_f1 = 0
            # best_threshold = 0.5 # found best treshold 0.52
            # thresholds = np.linspace(0.3, 0.7, 41)
            # for t in thresholds:
            #     preds = (np.array(val_probs) > t).astype(int)
            #     f1 = f1_score(all_labels, preds)
            #     if f1 > best_f1:
            #         best_f1 = f1
            #         best_threshold = t

            all_preds = (np.array(val_probs) > prob_treshold).astype(int)
            f1 = f1_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_preds)
            scheduler.step() #f1 - f1 should be used for ReduceLROnPlateau scheduler as argument

            if (epoch + 1)% 100 == 0:
                print(f"Epoch {epoch + 1}: F1 = {f1:.4f}, AUC = {auc:.4f}")
                # print(f"Best threshold: {best_threshold:.2f} with F1 = {best_f1:.4f}")
        self.save_model()

    def save_model(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/rnn_classifier_{timestamp}.pth"
        torch.save(self.state_dict(), model_path)
        print(f"RNN model saved to {model_path}")

    def load_model(self, model_path, device='cpu'):
        self.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        self.eval()
        print(f"RNN Classifier Model loaded from {model_path}")

    def test_rnn_model(self, test_x, test_y, prob_treshold=0.52):
        self.eval()
        test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y).float())
        test_loader = DataLoader(test_dataset, batch_size=32)

        all_preds, all_labels = [], []
        val_probs = []
        with torch.no_grad():
            for xb, yb in test_loader:
                pred, _ = self(xb)
                prob = torch.sigmoid(pred).numpy()
                val_probs.extend(prob)

                # preds = np.round(prob)
                # all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        all_preds = (np.array(val_probs) > prob_treshold).astype(int)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Test Results — F1: {f1:.4f}, AUC: {auc:.4f}")


class AttentionLSTM(RNNClassifier):
    def __init__(self, input_dim, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)
        self.lstm = torch_nn.LSTM(input_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attn = torch_nn.Linear(hidden_size, 1)
        self.dropout = torch_nn.Dropout(dropout)
        self.fc = torch_nn.Sequential(
            torch_nn.Linear(hidden_size, hidden_size // 2),
            torch_nn.ReLU(), # positive vals
            torch_nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):  # x: (batch, seq, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # (batch, seq, hidden) * (hidden, 1) -> (batch, seq, 1) -> (batch, seq)
        # context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, seq, hidden) .* (batch, seq, 1) = (batch, hidden) - sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        context = self.dropout(context) # (batch, hidden)
        output = self.fc(context)  # (batch_size, 1)
        return output.squeeze(-1), attn_weights