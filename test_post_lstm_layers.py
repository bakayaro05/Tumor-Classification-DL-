import torch
import torch.nn as nn


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 2. SIMULATED LSTM OUTPUT
# ============================================================

# This represents the final LSTM feature vector obtained from:
#
# lstm_output[:, -1, :]
#
# Shape:
# [batch_size, hidden_size]
#
# In our current implementation:
# [32, 256]

x = torch.randn(
    32,
    256,
    device=device
)

print("\nInput from LSTM:")
print(x.shape)


# ============================================================
# 3. BATCH NORMALIZATION
# ============================================================

batch_norm = nn.BatchNorm1d(
    num_features=256
).to(device)

x = batch_norm(x)

print("\nAfter BatchNorm:")
print(x.shape)


# ============================================================
# 4. DROPOUT
# ============================================================

# The base paper specifies dropout = 0.3

dropout = nn.Dropout(
    p=0.3
).to(device)

x = dropout(x)

print("\nAfter Dropout:")
print(x.shape)


# ============================================================
# 5. DENSE LAYER
# ============================================================

# The paper mentions Dense layers after the LSTM,
# but does not clearly specify the number of units.
#
# For our implementation we use:
#
# 256 → 128

dense = nn.Linear(
    in_features=256,
    out_features=128
).to(device)

x = dense(x)

print("\nAfter Dense:")
print(x.shape)


# ============================================================
# 6. ACTIVATION
# ============================================================

relu = nn.ReLU()

features = relu(x)

print("\nAfter ReLU:")
print(features.shape)


# ============================================================
# 7. VERIFY
# ============================================================

assert tuple(features.shape) == (32, 128)

print("\n✓ BatchNorm + Dropout + Dense shapes are correct")

print("\nFINAL DEEP FEATURE SHAPE:")
print(features.shape)

print("\nREADY TO BUILD COMPLETE RESNET152 + LSTM MODEL")