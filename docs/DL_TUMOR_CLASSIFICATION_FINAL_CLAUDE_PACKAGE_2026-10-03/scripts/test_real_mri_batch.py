import torch

from src.data_pipeline_torch import create_dataloaders
from model_resnet152_lstm import ResNet152LSTM


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. LOAD EXISTING DATA PIPELINE
# ============================================================

(
    base_train_dataset,
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader,
    train_indices,
    val_indices,
) = create_dataloaders()

print("\nDataLoaders created successfully")


# ============================================================
# 3. CLASS MAPPING
# ============================================================

print("\nClass mapping:")

for class_name, class_index in (
    base_train_dataset.class_to_idx.items()
):
    print(
        f"{class_index} -> {class_name}"
    )


# ============================================================
# 4. GET ONE REAL MRI TRAINING BATCH
# ============================================================

images, labels = next(
    iter(train_loader)
)

print("\nREAL TRAINING BATCH")

print("Images shape:")
print(images.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nImage dtype:")
print(images.dtype)

print("\nLabel dtype:")
print(labels.dtype)

print("\nLabel minimum:")
print(labels.min().item())

print("\nLabel maximum:")
print(labels.max().item())

print("\nLabels in this batch:")
print(labels.tolist())


# ============================================================
# 5. VERIFY REAL DATA BEFORE GPU
# ============================================================

assert images.ndim == 4
assert images.shape[1:] == (3, 224, 224)

assert labels.ndim == 1

assert labels.min().item() >= 0
assert labels.max().item() <= 3

assert labels.dtype == torch.int64

print("\n✓ Real MRI batch structure is correct")


# ============================================================
# 6. MOVE REAL BATCH TO GPU
# ============================================================

images = images.to(
    device,
    non_blocking=True,
)

labels = labels.to(
    device,
    non_blocking=True,
)

print("\nImages device:")
print(images.device)

print("\nLabels device:")
print(labels.device)


# ============================================================
# 7. BUILD COMPLETE MODEL
# ============================================================

model = ResNet152LSTM(
    lstm_hidden_size=256,
    dense_features=128,
    num_classes=4,
    dropout=0.3,
)

model = model.to(device)

model.eval()

print("\nComplete ResNet152 + LSTM model loaded")


# ============================================================
# 8. REAL MRI FORWARD PASS
# ============================================================

with torch.no_grad():

    logits = model(images)

    features = model(
        images,
        return_features=True,
    )


# ============================================================
# 9. OUTPUT SHAPES
# ============================================================

print("\nREAL MRI MODEL OUTPUT")

print("\nClassification logits:")
print(logits.shape)

print("\nDeep features:")
print(features.shape)


# ============================================================
# 10. VERIFY MODEL OUTPUT
# ============================================================

batch_size = images.shape[0]

assert logits.shape == (
    batch_size,
    4,
)

assert features.shape == (
    batch_size,
    128,
)

print(
    "\n✓ REAL MRI BATCH PASSED "
    "THROUGH COMPLETE MODEL"
)


# ============================================================
# 11. TEMPORARY PREDICTIONS
# ============================================================

predictions = torch.argmax(
    logits,
    dim=1,
)

print("\nTemporary predictions:")
print(predictions.tolist())

print("\nTrue labels:")
print(labels.tolist())


# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

print(
    "REAL DATA + RESNET152 + LSTM "
    "FORWARD PASS SUCCESSFUL"
)

print("=" * 70)

print(
    "\nNEXT STEP: "
    "prepare the actual training loop."
)