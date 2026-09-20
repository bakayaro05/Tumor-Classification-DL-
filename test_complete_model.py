import torch
from model_resnet152_lstm import ResNet152LSTM


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# Build complete model
# ------------------------------------------------------------

model = ResNet152LSTM(
    lstm_hidden_size=256,
    dense_features=128,
    num_classes=4,
    dropout=0.3
)

model = model.to(device)
model.eval()

print("\nComplete ResNet152 + LSTM model loaded")


# ------------------------------------------------------------
# Temporary synthetic MRI batch
# ------------------------------------------------------------

batch_size = 32

x = torch.randn(
    batch_size,
    3,
    224,
    224,
    device=device
)

print("\nInput:")
print(x.shape)


# ------------------------------------------------------------
# Full forward pass
# ------------------------------------------------------------

with torch.no_grad():

    logits = model(x)

    features = model(
        x,
        return_features=True
    )


print("\n4-class output:")
print(logits.shape)

print("\nDeep feature output:")
print(features.shape)


# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

assert logits.shape == (batch_size, 4)
assert features.shape == (batch_size, 128)

print("\n✓ COMPLETE MODEL FORWARD PASS SUCCESSFUL")

print("\nNext step:")
print("Pass one REAL MRI DataLoader batch through this model.")