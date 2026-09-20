import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class ResNet152LSTM(nn.Module):
    def __init__(
        self,
        lstm_hidden_size=256,
        dense_features=128,
        num_classes=4,
        dropout=0.3
    ):
        super().__init__()

        # ----------------------------------------------------
        # ResNet152 convolutional backbone
        # ----------------------------------------------------
        weights = ResNet152_Weights.IMAGENET1K_V2
        resnet = resnet152(weights=weights)

        # Remove adaptive avg pooling and final classifier
        self.cnn_backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        # ----------------------------------------------------
        # Post-LSTM layers
        # ----------------------------------------------------
        self.batch_norm = nn.BatchNorm1d(lstm_hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(
            lstm_hidden_size,
            dense_features
        )

        self.relu = nn.ReLU()

        # Temporary neural-network classifier
        self.classifier = nn.Linear(
            dense_features,
            num_classes
        )


    def forward(self, x, return_features=False):

        # ----------------------------------------------------
        # CNN feature extraction
        #
        # [B, 3, 224, 224]
        #       ↓
        # [B, 2048, 7, 7]
        # ----------------------------------------------------
        x = self.cnn_backbone(x)

        # ----------------------------------------------------
        # Convert spatial feature map into sequence
        #
        # [B, 2048, 7, 7]
        #       ↓
        # [B, 7, 7, 2048]
        #       ↓
        # [B, 49, 2048]
        # ----------------------------------------------------
        x = x.permute(0, 2, 3, 1)

        x = x.reshape(
            x.size(0),
            -1,
            x.size(-1)
        )

        # ----------------------------------------------------
        # LSTM
        #
        # [B, 49, 2048]
        #       ↓
        # [B, 49, 256]
        # ----------------------------------------------------
        lstm_output, _ = self.lstm(x)

        # Final sequence position
        x = lstm_output[:, -1, :]

        # ----------------------------------------------------
        # Post-LSTM feature block
        # ----------------------------------------------------
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.dense(x)

        features = self.relu(x)

        # Used later when extracting features for SVM
        if return_features:
            return features

        # Temporary 4-class NN output
        logits = self.classifier(features)

        return logits