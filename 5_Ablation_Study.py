import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

CSV_PATH = "data/MANIAC_benchmark_dataset.csv"
BATCH_SIZE = 4096
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path, drop_feats=[]):
    df = pd.read_csv(path)
    df['diff'] = df['frame_start_x'].diff().fillna(0)
    df['video_id'] = (df['diff'] < 0).cumsum()
    
    feats = {
        'Interaction': ['avg_distance', 'var_distance', 'contact_count', 'contact_duration', 'distance_trend'],
        'Kinematics': ['avg_speed', 'acceleration', 'avg_angle']
    }
    df['distance_trend'] = df['distance_trend'].map({'increasing': 1, 'decreasing': 0}).fillna(0)
    
    selected = []
    if 'Interaction' not in drop_feats: selected.extend(feats['Interaction'])
    if 'Kinematics' not in drop_feats: selected.extend(feats['Kinematics'])
    if 'All_Except_Distance' in drop_feats: selected = ['avg_distance']
        
    X = df[selected].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['label'].values)
    groups = df['video_id'].values 
    X = StandardScaler().fit_transform(X)
    return X, y, groups, len(selected)

class BiRNN_Static(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(128 * 2, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def run_experiment(name, drop_list):
    X, y, groups, input_dim = load_data(CSV_PATH, drop_list)
    gkf = GroupKFold(n_splits=5)
    trues, preds = [], []
    
    for tr, te in gkf.split(X, y, groups):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        class DS(Dataset):
            def __init__(self, x, y): self.x, self.y = torch.tensor(x), torch.tensor(y).long()
            def __len__(self): return len(self.x)
            def __getitem__(self, i): return self.x[i], self.y[i]
        
        model = BiRNN_Static(input_dim, 5).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in DataLoader(DS(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
                
        model.eval()
        with torch.no_grad():
            for xb, yb in DataLoader(DS(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False):
                trues.extend(yb.numpy())
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                
    acc = accuracy_score(trues, preds)
    gf1 = f1_score(trues, preds, average=None)[1] # 1 is grabbing index usually
    return {"Config": name, "Overall Acc": acc, "Grabbing F1": gf1}

results = []
results.append(run_experiment("Full Features", []))
results.append(run_experiment("No Kinematics", ['Kinematics']))
results.append(run_experiment("No Interaction", ['Interaction']))
results.append(run_experiment("Distance Only", ['All_Except_Distance']))
print(pd.DataFrame(results))
