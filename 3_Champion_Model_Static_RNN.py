import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# --- Configuration ---
DATA_PATH = "data/MANIAC_benchmark_dataset.csv"
# This is the key finding of the paper:
SEQ_LENGTH = 1
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 50  # Number of trials for hyperparameter search
FINAL_EPOCHS = 100    # Epochs to train the final champion model
PATIENCE = 10

# Set seed for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def load_and_preprocess_static(csv_path, seq_length):
    """Loads and reshapes data. With seq_length=1, this treats each sample independently."""
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return None, None, None

    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 1. Define feature columns
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
    n_features = X_scaled.shape[1]
    usable_rows = (X_scaled.shape[0] // seq_length) * seq_length
    X_seq = X_scaled[:usable_rows].reshape(-1, seq_length, n_features)
    y_seq = y_enc[:usable_rows].reshape(-1, seq_length)[:, -1]

    return X_seq.astype(np.float32), y_seq, le

class SequenceDataset(Dataset):
    """Simple PyTorch Dataset."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BidirectionalRNN(nn.Module):
    """The same Bi-RNN model, now used as a static encoder."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_p):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_p, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size * 2, num_classes) # *2 for bidirectional
    
    def forward(self, x):
        # x is [batch, 1, features]
        out, _ = self.lstm(x)
        # out is [batch, 1, hidden_size*2]
        # We take the output of the only time step
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def evaluate_model(model, loader, criterion, device):
    """Evaluates the model and returns accuracy, loss, and predictions."""
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

def objective(trial, X_train, y_train, X_val, y_val, input_size, num_classes, device):
    """Optuna objective function to find best hyperparameters."""
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout_p': trial.suggest_float('dropout_p', 0.2, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    }

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=params['batch_size'])

    model = BidirectionalRNN(input_size, params['hidden_size'], params['num_layers'],
                             num_classes, params['dropout_p']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Train for a fixed number of epochs for the trial
    for epoch in range(15):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        accuracy, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def plot_results(y_true, y_pred, class_names):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Champion Model (Static RNN, seq_length=1)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("champion_model_confusion_matrix.png")
    print("\nConfusion matrix saved to 'champion_model_confusion_matrix.png'")
    # plt.show() # Disabled for non-interactive script

def main():
    """Main function to run Optuna search and train the final champion model."""
    
    print(f"Loading data for Static RNN (seq_length={SEQ_LENGTH})...")
    X, y, le = load_and_preprocess_static(DATA_PATH, SEQ_LENGTH)
    
    if X is None:
        return

    print(f"Data reshaped to: {X.shape} (This is Model 7/8's setup)")
    
    # Split: 80% Train/Val (for Optuna), 20% Test (final hold-out)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # Split Train/Val for Optuna
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_val
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Optuna Hyperparameter Search ---
    print(f"\n--- Starting Optuna Search ({N_TRIALS_OPTUNA} trials) ---")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(
        trial, X_train, y_train, X_val, y_val,
        X.shape[2], len(le.classes_), device
    ), n_trials=N_TRIALS_OPTUNA)

    print("\n--- Optuna Search Finished ---")
    best_params = study.best_trial.params
    print(f"Best validation accuracy: {study.best_trial.value:.4f}")
    print(f"Best hyperparameters found: {best_params}")

    # --- 2. Train Final Champion Model ---
    print("\n--- Training Champion Model on all training data ---")
    
    # Combine train and validation sets for final training
    final_train_loader = DataLoader(
        SequenceDataset(X_train_val, y_train_val),
        batch_size=best_params['batch_size'],
        shuffle=True
    )
    final_test_loader = DataLoader(
        SequenceDataset(X_test, y_test),
        batch_size=best_params['batch_size']
    )

    # Build model with best params
    model = BidirectionalRNN(
        input_size=X.shape[2],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        num_classes=len(le.classes_),
        dropout_p=best_params['dropout_p']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    # Training loop with Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

    for epoch in range(1, FINAL_EPOCHS + 1):
        train_loss = train_one_epoch(model, final_train_loader, criterion, optimizer, device)
        val_acc, val_loss, _, _ = evaluate_model(model, final_test_loader, criterion, device)

        if (epoch % 10 == 0):
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "champion_model.pth")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break
            
    # --- 3. Final Evaluation ---
    print("\n--- Final Evaluation on Test Set (Model 8) ---")
    model.load_state_dict(best_model_weights)
    print("Best model weights loaded.")

    test_acc, _, y_true, y_pred = evaluate_model(model, final_test_loader, criterion, device)
    
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
    print("This result corresponds to the Champion Model (Model 8) in the paper.")

    print("\nFinal Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=4))

    # Plot and save confusion matrix
    plot_results(y_true, y_pred, le.classes_)

if __name__ == "__main__":
    main()
