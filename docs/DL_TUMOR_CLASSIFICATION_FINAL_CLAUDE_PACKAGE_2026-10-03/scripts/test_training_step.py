import torch
import torch.nn as nn

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
# 2. LOAD EXISTING REAL DATA PIPELINE
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


# ============================================================
# 3. GET ONE REAL TRAINING BATCH
# ============================================================

images, labels = next(iter(train_loader))


# ------------------------------------------------------------
# Training uses more GPU memory than inference.
#
# For this sanity test we only use 8 real MRI images.
# This does NOT change the dataset or saved split.
# ------------------------------------------------------------

SANITY_BATCH_SIZE = 8

images = images[:SANITY_BATCH_SIZE]
labels = labels[:SANITY_BATCH_SIZE]


images = images.to(
    device,
    non_blocking=True
)

labels = labels.to(
    device,
    non_blocking=True
)


print("\nTraining sanity batch:")

print("Images:", images.shape)
print("Labels:", labels.shape)


# ============================================================
# 4. BUILD MODEL
# ============================================================

model = ResNet152LSTM(
    lstm_hidden_size=256,
    dense_features=128,
    num_classes=4,
    dropout=0.3,
)

model = model.to(device)

# VERY IMPORTANT:
# Training mode enables Dropout and BatchNorm training behavior.
model.train()


# ============================================================
# 5. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. OPTIMIZER
# ============================================================

# RMSprop is being used for the ResNet152 experiment.
#
# Learning rate here is currently an implementation choice
# unless/until we confirm a different explicit value from
# the paper.

optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=1e-4,
)


# ============================================================
# 7. ZERO EXISTING GRADIENTS
# ============================================================

optimizer.zero_grad(
    set_to_none=True
)


# ============================================================
# 8. FORWARD PASS
# ============================================================

logits = model(images)

print("\nLogits shape:")
print(logits.shape)


# ============================================================
# 9. COMPUTE LOSS
# ============================================================

loss = criterion(
    logits,
    labels
)

print("\nLoss before backward:")
print(loss.item())


# ============================================================
# 10. VERIFY LOSS
# ============================================================

assert torch.isfinite(loss).item(), (
    "Loss is NaN or infinity."
)

print("\n✓ Loss is finite")


# ============================================================
# 11. BACKPROPAGATION
# ============================================================

loss.backward()

print("\n✓ Backpropagation completed")


# ============================================================
# 12. CHECK THAT GRADIENTS EXIST
# ============================================================

gradient_found = False

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        if torch.isfinite(parameter.grad).all():

            gradient_found = True
            break


assert gradient_found, (
    "No valid model gradients were found."
)

print("✓ Valid gradients detected")


# ============================================================
# 13. OPTIMIZER STEP
# ============================================================

optimizer.step()

print("✓ RMSprop optimizer step completed")


# ============================================================
# 14. TEMPORARY TRAINING ACCURACY
# ============================================================

predictions = torch.argmax(
    logits,
    dim=1
)

correct = (
    predictions == labels
).sum().item()

accuracy = (
    correct / labels.size(0)
) * 100.0


print("\nPredictions:")
print(predictions.tolist())

print("\nTrue labels:")
print(labels.tolist())

print(
    f"\nTemporary one-batch accuracy: "
    f"{accuracy:.2f}%"
)


# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

print(
    "ONE REAL TRAINING STEP "
    "COMPLETED SUCCESSFULLY"
)

print("=" * 70)

print(
    "\nThe complete ResNet152 + LSTM model "
    "can forward-propagate, calculate loss, "
    "backpropagate gradients and update weights."
)

print(
    "\nNEXT STEP: build the full "
    "training + validation loop."
)