"""
ECG Arrhythmia Classification - Training Script
Dataset: MIT-BIH Arrhythmia Dataset
"""

import pandas as pd
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------

DATA_PATH = "/kaggle/input/datasets/shayanfazeli/heartbeat/mitbih_train.csv"

train_df = pd.read_csv(DATA_PATH, header=None)

# ---------------------------------------------------------------------
# 2. Handle Class Imbalance
# ---------------------------------------------------------------------

df_0 = train_df[train_df[187] == 0]
df_1 = train_df[train_df[187] == 1]
df_2 = train_df[train_df[187] == 2]
df_3 = train_df[train_df[187] == 3]
df_4 = train_df[train_df[187] == 4]


def upsample_class(df):
    return resample(
        df,
        replace=True,
        n_samples=20000,
        random_state=42
    )


balanced_df = pd.concat([
    df_0.sample(n=20000, random_state=42),
    upsample_class(df_1),
    upsample_class(df_2),
    upsample_class(df_3),
    upsample_class(df_4)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------------------
# 3. Prepare Data
# ---------------------------------------------------------------------

X = balanced_df.iloc[:, :186].values
y = balanced_df.iloc[:, 187].values

X = X.reshape((-1, 186, 1))

# ---------------------------------------------------------------------
# 4. Build 1D CNN
# ---------------------------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(186, 1)),
    layers.Conv1D(64, 6, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=3, strides=2),

    layers.Conv1D(64, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2, strides=2),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(5, activation="softmax")
])

# ---------------------------------------------------------------------
# 5. Compile & Train
# ---------------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
    verbose=1
)

# ---------------------------------------------------------------------
# 6. Save Model
# ---------------------------------------------------------------------

model.save("ecg_model.h5")

print("\nTraining Complete!")
print("Model saved as ecg_model.h5")
