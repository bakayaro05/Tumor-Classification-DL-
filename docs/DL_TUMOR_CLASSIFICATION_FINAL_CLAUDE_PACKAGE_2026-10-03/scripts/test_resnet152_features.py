import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 2. LOAD PRETRAINED RESNET152
# ============================================================

weights = ResNet152_Weights.IMAGENET1K_V2

resnet = resnet152(weights=weights)

print("\nFull ResNet152 loaded.")


# ============================================================
# 3. REMOVE AVGPOOL + FULLY CONNECTED CLASSIFIER
# ============================================================

# Original ResNet152 ends like this:
#
# convolutional layers
#       ↓
# layer4
#       ↓
# AdaptiveAvgPool2d
#       ↓
# Linear classifier
#
# We ONLY want the convolutional feature map.
#
# children() gives all major ResNet blocks.
# [:-2] removes:
#
#   AdaptiveAvgPool2d
#   Linear layer
#

cnn_backbone = nn.Sequential(
    *list(resnet.children())[:-2]
)

cnn_backbone = cnn_backbone.to(device)

# We are using ResNet initially as a pretrained feature extractor.
cnn_backbone.eval()


# ============================================================
# 4. CREATE ONE TEST BATCH
# ============================================================

# IMPORTANT:
# This temporary tensor simply checks the network architecture.
#
# Your real DataLoader batch will have the same shape:
#
# [batch_size, channels, height, width]
#
# Expected:
# [32, 3, 224, 224]

x = torch.randn(
    32,
    3,
    224,
    224,
    device=device
)

print("\nInput batch shape:")
print(x.shape)


# ============================================================
# 5. PASS THROUGH RESNET152 CONVOLUTIONAL BACKBONE
# ============================================================

with torch.no_grad():
    features = cnn_backbone(x)


print("\nResNet152 convolutional feature shape:")
print(features.shape)


# ============================================================
# 6. VERIFY EXPECTED FEATURE MAP
# ============================================================

expected_feature_shape = (32, 2048, 7, 7)

assert tuple(features.shape) == expected_feature_shape, (
    f"Unexpected feature shape: {features.shape}. "
    f"Expected: {expected_feature_shape}"
)

print("\n✓ ResNet152 feature map is correct")


# ============================================================
# 7. RESHAPE FEATURE MAP FOR LSTM
# ============================================================

# Current shape:
#
# [32, 2048, 7, 7]
#
# LSTM expects:
#
# [batch, sequence_length, input_features]
#
# We interpret the 7×7 spatial grid as 49 sequential positions.
#
# 7 × 7 = 49
#
# Each spatial position contains a 2048-dimensional feature vector.
#
# Desired:
#
# [32, 49, 2048]


# Move the channel dimension to the end:
#
# [32, 2048, 7, 7]
#       ↓
# [32, 7, 7, 2048]

features = features.permute(0, 2, 3, 1)

print("\nAfter permute:")
print(features.shape)


# Flatten the 7×7 spatial dimensions:
#
# [32, 7, 7, 2048]
#       ↓
# [32, 49, 2048]

sequence_features = features.reshape(
    features.size(0),
    -1,
    features.size(-1)
)

print("\nLSTM input shape:")
print(sequence_features.shape)


# ============================================================
# 8. VERIFY EXPECTED LSTM INPUT SHAPE
# ============================================================

expected_lstm_shape = (32, 49, 2048)

assert tuple(sequence_features.shape) == expected_lstm_shape, (
    f"Unexpected LSTM input shape: {sequence_features.shape}. "
    f"Expected: {expected_lstm_shape}"
)

print("\n✓ ResNet152 → LSTM reshaping successful")
print("\nREADY FOR LSTM")