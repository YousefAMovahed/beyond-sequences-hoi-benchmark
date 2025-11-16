import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import copy

# --- Configuration ---
DATA_PATH = "data/MANIAC_benchmark_dataset.csv"
SEQ_LENGTH = 5  # This is the key parameter for the "Temporal Hypothesis"
N_SPLITS = 5
BATCH_SIZE = 64
EPOCHS_PER_FOLD = 40
PATIENCE = 7
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

# Set seed for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def load_and_preprocess_temporal(csv_path, seq_length):
    """Loads and reshapes the data into temporal sequences."""
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return None, None, None

    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 1. Define feature columns (same as MLP, but will be sequenced)
    drop_cols = ['sample_id', 'label_frame', 'frame_start_x', 'frame_end_x',
                 'frame_start_y', 'frame_end_y', 'label']
    df_features = df.drop(columns=drop_cols, errors='ignore')
    df_features = pd.get_dummies(df_features, columns=['distance_trend'], prefix='trend')
    
    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # 3. Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(df['label'].values)

    # 4. Reshape into sequences
    # We truncate the data to be divisible by seq_length
    n_features = X_scaled.shape[1]
    usable_rows = (X_scaled.shape[0] // seq_length) * seq_length
    X_seq = X_scaled[:usable_rows].reshape(-1, seq_length, n_features)
    
    # Get the label for the *last* item in each sequence
    y_seq = y_enc[:usable_rows].reshape(-1, seq_length)[:, -1]

    return X_seq.astype(np.float32), y_seq, le

class SequenceDataset(Dataset):
    """Simple PyTorch Dataset for sequence data."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BidirectionalRNN(nn.Module):
    """PyTorch Bi-RNN model."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_p):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_p, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size * 2, num_classes) # *2 for bidirectional
    
    def forward(self, x):
        out, _ = self.lstm(x)
        # Get the output of the last time step
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(loader.dataset)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    accuracy = (y_pred == y_true).mean()
    avg_loss = running_loss / len(loader.dataset)
    return accuracy, avg_loss, y_true, y_pred

def main():
    """Main function to run K-Fold CV for the Temporal Bi-RNN model."""
    
    print(f"Loading and preprocessing data for Temporal Bi-RNN (seq_length={SEQ_LENGTH})...")
    X, y, le = load_and_preprocess_temporal(DATA_PATH, SEQ_LENGTH)

    if X is None:
        return

    print(f"Data reshaped to: {X.shape}")
    print(f"Labels reshaped to: {y.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    all_true_labels = []
    all_pred_labels = []
    fold_accuracies = []

    print(f"\n--- Starting {N_SPLITS}-Fold Cross-Validation for Temporal Bi-RNN (Model 6) ---")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X, y), 1):
        print(f"--- Fold {fold}/{N_SPLITS} ---")
        
        # 1. Create datasets and dataloaders for this fold
        train_dataset = SequenceDataset(X[train_ids], y[train_ids])
        val_dataset = SequenceDataset(X[val_ids], y[val_ids])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 2. Build new model instance
        model = BidirectionalRNN(
            input_size=X.shape[2],
            hidden_size=128,
            num_layers=2,
            num_classes=len(le.classes_),
            dropout_p=0.4
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 3. Training loop with simple Early Stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(1, EPOCHS_PER_FOLD + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
            
            if (epoch % 10 == 0):
                print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break
        
        # 4. Load best model and evaluate on validation set
        model.load_state_dict(best_model_weights)
        
        val_acc, _, y_true_fold, y_pred_fold = evaluate_model(model, val_loader, criterion, device)
        print(f"Fold {fold} Final Accuracy: {val_acc * 100:.2f}%")
        
        all_true_labels.extend(y_true_fold)
        all_pred_labels.extend(y_pred_fold)
        fold_accuracies.append(val_acc)
    
    # --- Final Report ---
    print("\n\n--- Temporal Bi-RNN K-Fold Results (Model 6) ---")
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"Mean Accuracy: {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)")
    print("This result corresponds to Model 6 in the paper (the temporal hypothesis).")
    
    print("\nOverall Classification Report (Aggregated across all folds):")
    print(classification_report(all_true_labels, all_pred_labels, target_names=le.classes_))


if __name__ == "__main__":
    main()
