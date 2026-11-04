import torch
import torch.nn as nn


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 2. TEST INPUT
# ============================================================

# This represents the output from ResNet152 after reshaping:
#
# [batch_size, sequence_length, input_features]
#
# [32, 49, 2048]

x = torch.randn(
    32,
    49,
    2048,
    device=device
)

print("\nInput to LSTM:")
print(x.shape)


# ============================================================
# 3. DEFINE LSTM
# ============================================================

# input_size = 2048
# because each of the 49 sequence positions contains
# a 2048-dimensional ResNet feature vector.
#
# hidden_size = 256
# This gives the LSTM a compact learned representation.
#
# batch_first=True means:
#
# [batch, sequence, features]
#
# rather than:
#
# [sequence, batch, features]

lstm = nn.LSTM(
    input_size=2048,
    hidden_size=256,
    num_layers=1,
    batch_first=True
).to(device)


# ============================================================
# 4. FORWARD PASS
# ============================================================

with torch.no_grad():

    lstm_output, (hidden_state, cell_state) = lstm(x)


print("\nFull LSTM output:")
print(lstm_output.shape)

print("\nHidden state:")
print(hidden_state.shape)

print("\nCell state:")
print(cell_state.shape)


# ============================================================
# 5. TAKE FINAL SEQUENCE OUTPUT
# ============================================================

# lstm_output shape:
#
# [32, 49, 256]
#
# We take the final sequence position:
#
# lstm_output[:, -1, :]
#
# Result:
#
# [32, 256]

final_features = lstm_output[:, -1, :]

print("\nFinal LSTM feature vector:")
print(final_features.shape)


# ============================================================
# 6. VERIFY
# ============================================================

assert tuple(lstm_output.shape) == (32, 49, 256)

assert tuple(final_features.shape) == (32, 256)

print("\n✓ LSTM output shapes are correct")

print("\nREADY FOR BATCHNORM + DROPOUT + DENSE")