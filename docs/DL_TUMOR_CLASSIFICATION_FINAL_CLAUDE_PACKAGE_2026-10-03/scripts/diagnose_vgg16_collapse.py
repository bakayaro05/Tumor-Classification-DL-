from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_cnn_lstm import CNNLSTM
from src.data_pipeline_torch import create_dataloaders


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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


images, labels = next(iter(train_loader))

images = images[:8].to(device)
labels = labels[:8].to(device)


model = CNNLSTM(
    backbone_name="vgg16",
    lstm_hidden_size=256,
    dense_features=128,
    num_classes=4,
    dropout=0.3,
).to(device)


model.eval()


with torch.no_grad():

    debug = model(
        images,
        return_debug=True,
    )


def stats(name, tensor):

    print(f"\n{name}")

    print("Shape :", tuple(tensor.shape))
    print("Mean  :", tensor.mean().item())
    print("Std   :", tensor.std().item())
    print("Min   :", tensor.min().item())
    print("Max   :", tensor.max().item())

    print(
        "Nonzero fraction:",
        (tensor != 0).float().mean().item()
    )


print("=" * 70)
print("VGG16 PRE-TRAINING ACTIVATION DIAGNOSTIC")
print("=" * 70)


stats(
    "CNN FEATURE MAP",
    debug["cnn_feature_map"],
)

stats(
    "LSTM SEQUENCE",
    debug["lstm_sequence"],
)

stats(
    "LSTM OUTPUT",
    debug["lstm_output"],
)

stats(
    "DEEP FEATURES",
    debug["deep_features"],
)


print("\nLabels:")
print(labels.tolist())