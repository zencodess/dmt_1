import datetime
import torch
import torch.nn as torch_nn
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import torch
import torch.nn as nn



class RNNClassifier(torch_nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.rnn = torch_nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dense = torch_nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dense(out[:, -1, :])
        return out.squeeze(), 0

    def train_rnn_model(self, train_x, train_y, val_x, val_y, device='cpu', epochs=500, batch_size=32, lr=0.005, prob_treshold=0.52):
        self.to(device)
        # print("Train label balance:", np.bincount(train_y))
        # print("Val label balance:", np.bincount(val_y))

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
                xb, yb = xb.to(device), yb.to(device)
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
                    xb = xb.to(device)
                    pred, _ = self(xb)
                    prob = torch.sigmoid(pred).cpu().numpy()
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
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.eval()
        self.to(device)
        print(f"RNN Classifier Model loaded from {model_path}")

    def test_rnn_model(self, test_x, test_y, device='cpu', prob_treshold=0.52):
        self.eval()
        self.to(device)

        test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y).float())
        test_loader = DataLoader(test_dataset, batch_size=32)

        all_preds, all_labels = [], []
        val_probs = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred, _ = self(xb)
                prob = torch.sigmoid(pred).cpu().numpy()
                val_probs.extend(prob)

                # preds = np.round(prob)
                # all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        all_preds = (np.array(val_probs) > prob_treshold).astype(int)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Test Results — F1: {f1:.4f}, AUC: {auc:.4f}")

# class AttentionLSTM(RNNClassifier):
#     def __init__(self, input_dim, hidden_size=128):
#         super().__init__(input_dim=input_dim)
#         self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
#         self.attn = nn.Linear(hidden_size, 1)  # attention over time
#         self.fc = nn.Linear(hidden_size, 1)  # final prediction
#
#     def forward(self, x):  # x: (batch, time, features)
#         lstm_out, _ = self.lstm(x)  # lstm_out: (batch, time, hidden)
#         attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # (batch, time)
#         context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden)
#         output = self.fc(context)
#         return output.squeeze(-1), attn_weights

class AttentionLSTM(RNNClassifier):
    def __init__(self, input_dim, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_size, num_layers=num_layers, dropout=dropout)
        self.feature_attn = torch_nn.Sequential(
            torch_nn.Linear(input_dim, input_dim),
            torch_nn.Tanh(),
            torch_nn.Softmax(dim=-1)
        )

        self.lstm = torch_nn.LSTM(input_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attn = torch_nn.Linear(hidden_size, 1)
        self.dropout = torch_nn.Dropout(dropout)
        self.fc = torch_nn.Sequential(
            torch_nn.Linear(hidden_size, hidden_size // 2),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):  # x: (batch, time, features)
        # Feature-level attention
        # feature_weights = self.feature_attn(x)  # (batch, time, features)
        # x = x * feature_weights

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden)

        # Temporal attention
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # (batch, time)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden)

        # Final prediction
        context = self.dropout(context)
        output = self.fc(context)
        return output.squeeze(-1), attn_weights