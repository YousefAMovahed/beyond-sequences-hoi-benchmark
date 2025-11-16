import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
DATA_PATH = "data/MANIAC_benchmark_dataset.csv"
N_SPLITS = 5
RANDOM_STATE = 42

def load_and_preprocess_data(csv_path):
    """Loads and preprocesses the static feature dataset for the MLP."""
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        print("Please download the dataset and place it in the 'data/' directory.")
        return None, None, None

    df = pd.read_csv(csv_path)

    # 1. Drop unused columns (as defined in the paper's methodology)
    drop_cols = ['sample_id', 'label_frame', 'frame_start_x', 'frame_end_x',
                 'frame_start_y', 'frame_end_y']
    df = df.drop(columns=drop_cols, errors='ignore')

    # 2. One-hot encode the categorical 'distance_trend' feature
    df = pd.get_dummies(df, columns=['distance_trend'], prefix='trend')

    # 3. Separate features (X) and labels (y)
    X = df.drop(columns=['label'])
    y = df['label']

    # 4. Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 5. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_enc, le.classes_

def build_optimized_mlp(input_dim, n_classes, l2_reg=1e-4, dropout_rate=0.4):
    """Builds the Keras MLP model (corresponds to Model 2/3 in the paper)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate / 2),

        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    """Main function to run the K-Fold cross-validation for the MLP baseline."""
    
    print("Loading and preprocessing data for MLP baseline...")
    X_scaled, y_enc, class_names = load_and_preprocess_data(DATA_PATH)
    
    if X_scaled is None:
        return

    print(f"Data loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"Target classes: {class_names}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_accuracies = []
    fold = 1

    print(f"\n--- Starting {N_SPLITS}-Fold Cross-Validation for Optimized MLP (Model 3) ---")

    for train_idx, test_idx in skf.split(X_scaled, y_enc):
        print(f"--- Fold {fold}/{N_SPLITS} ---")
        
        # 1. Split data for this fold
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        # 2. Compute class weights for this fold
        cw = dict(enumerate(compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )))

        # 3. Build a new model instance
        model = build_optimized_mlp(
            input_dim=X_train.shape[1],
            n_classes=len(class_names)
        )

        # 4. Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
        ]

        # 5. Train the model (using 10% of train data as validation)
        model.fit(
            X_train, y_train,
            validation_split=0.1, 
            epochs=100,
            batch_size=32,
            class_weight=cw,
            callbacks=callbacks,
            verbose=0
        )

        # 6. Evaluate on the hold-out test set for this fold
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold} Accuracy: {acc * 100:.2f}%")
        fold_accuracies.append(acc)
        fold += 1

    # --- Final Report ---
    print("\n--- MLP Baseline K-Fold Results ---")
    print(f"Mean Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")
    print(f"Std Deviation: {np.std(fold_accuracies) * 100:.2f}%")
    print("This result corresponds to Model 3 in the paper's results table.")

if __name__ == "__main__":
    main()
