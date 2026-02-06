import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

CSV_PATH = "data/MANIAC_benchmark_dataset.csv"
BATCH_SIZE = 2048
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path):
    df = pd.read_csv(path)
    df['diff'] = df['frame_start_x'].diff().fillna(0)
    df['video_id'] = (df['diff'] < 0).cumsum()
    feature_cols = ['avg_distance', 'var_distance', 'contact_count', 'contact_duration', 'avg_speed', 'acceleration', 'avg_angle']
    df['distance_trend'] = df['distance_trend'].map({'increasing': 1, 'decreasing': 0}).fillna(0)
    feature_cols.append('distance_trend')
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values
    groups = df['video_id'].values 
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, groups, le, len(feature_cols)

def create_temporal_data(X, y, groups, seq_len=5):
    X_seq, y_seq, g_seq = [], [], []
    unique_groups = np.unique(groups)
    for grp in unique_groups:
        indices = np.where(groups == grp)[0]
        if len(indices) < seq_len: continue
        for i in range(len(indices) - seq_len + 1):
            X_seq.append(X[indices[i : i + seq_len]])
            y_seq.append(y[indices[i + seq_len - 1]])
            g_seq.append(grp)
    return np.array(X_seq), np.array(y_seq), np.array(g_seq)

class MLP_Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    def forward(self, x): return self.net(x)

class BiLSTM_Net(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(128 * 2, num_classes) 
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def get_metrics(y_true, y_pred, le, model_name):
    grab_idx = le.transform(['grabbing'])[0]
    return {
        "Model": model_name,
        "Overall Acc.": accuracy_score(y_true, y_pred),
        "Weighted F1": f1_score(y_true, y_pred, average='weighted'),
        "Grabbing F1": f1_score(y_true, y_pred, average=None)[grab_idx]
    }

def train_eval_nn(model_class, input_dim, num_classes, X_tr, y_tr, X_te, y_te, seq_len=1):
    class DS(Dataset):
        def __init__(self, x, y): self.x, self.y = torch.tensor(x), torch.tensor(y).long()
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return self.x[i], self.y[i]
    train_dl = DataLoader(DS(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(DS(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)
    
    if model_class == MLP_Net: model = model_class(input_dim, num_classes).to(DEVICE)
    else: model = model_class(input_dim, num_classes, seq_len).to(DEVICE)
        
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(DEVICE)
            preds.extend(model(xb).argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    return np.array(trues), np.array(preds)

X, y, groups, le, n_feats = load_data(CSV_PATH)
num_classes = len(le.classes_)
gkf = GroupKFold(n_splits=5)
X_seq5, y_seq5, g_seq5 = create_temporal_data(X, y, groups, seq_len=5)

results = {'SVM':[], 'RF':[], 'MLP':[], 'Bi-RNN (Seq=1)':[], 'Bi-RNN (Seq=5)':[]}
trues = {k:[] for k in results}; preds = {k:[] for k in results}

print("Running Rigorous Group K-Fold Benchmark...")
for (tr, te), (tr5, te5) in tqdm(zip(gkf.split(X, y, groups), gkf.split(X_seq5, y_seq5, g_seq5)), total=5):
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    
    svm = SVC(kernel='rbf', C=1.0); svm.fit(X_tr, y_tr)
    trues['SVM'].extend(y_te); preds['SVM'].extend(svm.predict(X_te))
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1); rf.fit(X_tr, y_tr)
    trues['RF'].extend(y_te); preds['RF'].extend(rf.predict(X_te))
    
    yt, yp = train_eval_nn(MLP_Net, n_feats, num_classes, X_tr, y_tr, X_te, y_te)
    trues['MLP'].extend(yt); preds['MLP'].extend(yp)
    
    yt, yp = train_eval_nn(BiLSTM_Net, n_feats, num_classes, X_tr, y_tr, X_te, y_te, seq_len=1)
    trues['Bi-RNN (Seq=1)'].extend(yt); preds['Bi-RNN (Seq=1)'].extend(yp)
    
    X5_tr, X5_te, y5_tr, y5_te = X_seq5[tr5], X_seq5[te5], y_seq5[tr5], y_seq5[te5]
    yt, yp = train_eval_nn(BiLSTM_Net, n_feats, num_classes, X5_tr, y5_tr, X5_te, y5_te, seq_len=5)
    trues['Bi-RNN (Seq=5)'].extend(yt); preds['Bi-RNN (Seq=5)'].extend(yp)

final_table = []
for m in trues:
    final_table.append(get_metrics(trues[m], preds[m], le, m))
print(pd.DataFrame(final_table))
