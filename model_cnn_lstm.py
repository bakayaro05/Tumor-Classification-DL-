import torch
import torch.nn as nn

from torchvision.models import (
    vgg16,
    vgg19,
    resnet50,
    resnet152,
    efficientnet_b3,

    VGG16_Weights,
    VGG19_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    EfficientNet_B3_Weights,
)


# ============================================================
# SUPPORTED BACKBONES
# ============================================================

SUPPORTED_BACKBONES = [

    "vgg16",

    "vgg19",

    "resnet50",

    "resnet152",

    "efficientnet_b3",
]


# ============================================================
# GENERIC CNN + LSTM MODEL
# ============================================================

class CNNLSTM(nn.Module):

    def __init__(
        self,
        backbone_name,
        lstm_hidden_size=256,
        dense_features=128,
        num_classes=4,
        dropout=0.3,
    ):

        super().__init__()


        # ----------------------------------------------------
        # CHECK BACKBONE NAME
        # ----------------------------------------------------

        if backbone_name not in SUPPORTED_BACKBONES:

            raise ValueError(

                f"Unsupported backbone: "
                f"{backbone_name}. "

                f"Supported backbones: "
                f"{SUPPORTED_BACKBONES}"
            )


        self.backbone_name = backbone_name

        self.lstm_hidden_size = lstm_hidden_size

        self.dense_features = dense_features

        self.num_classes = num_classes

        self.dropout_probability = dropout


        # ====================================================
        # CNN BACKBONE
        # ====================================================

        (
            self.cnn_backbone,
            self.cnn_output_channels,
        ) = self._create_backbone(
            backbone_name
        )


        # ====================================================
        # LSTM
        # ====================================================

        # Each spatial position from the CNN becomes
        # one sequence element.
        #
        # Example:
        #
        # ResNet152:
        # [B, 2048, 7, 7]
        #
        # becomes:
        #
        # [B, 49, 2048]
        #
        # Therefore input_size must equal the number of
        # CNN output channels.

        self.lstm = nn.LSTM(

            input_size=self.cnn_output_channels,

            hidden_size=lstm_hidden_size,

            num_layers=1,

            batch_first=True,
        )


        # ====================================================
        # POST-LSTM BLOCK
        # ====================================================

        self.batch_norm = nn.BatchNorm1d(
            lstm_hidden_size
        )


        self.dropout = nn.Dropout(
            p=dropout
        )


        self.dense = nn.Linear(

            lstm_hidden_size,

            dense_features,
        )


        self.relu = nn.ReLU()


        # ====================================================
        # TEMPORARY NEURAL CLASSIFIER
        # ====================================================

        # This classifier is used to train the CNN + LSTM.
        #
        # Later the 128-dimensional Dense representation
        # is extracted for PCA + SVM.

        self.classifier = nn.Linear(

            dense_features,

            num_classes,
        )


    # ========================================================
    # CREATE PRETRAINED CNN BACKBONE
    # ========================================================

    def _create_backbone(
        self,
        backbone_name,
    ):


        # ----------------------------------------------------
        # VGG16
        # ----------------------------------------------------

        if backbone_name == "vgg16":

            model = vgg16(

                weights=(
                    VGG16_Weights.IMAGENET1K_V1
                )
            )


            backbone = model.features


            output_channels = 512


        # ----------------------------------------------------
        # VGG19
        # ----------------------------------------------------

        elif backbone_name == "vgg19":

            model = vgg19(

                weights=(
                    VGG19_Weights.IMAGENET1K_V1
                )
            )


            backbone = model.features


            output_channels = 512


        # ----------------------------------------------------
        # RESNET50
        # ----------------------------------------------------

        elif backbone_name == "resnet50":

            model = resnet50(

                weights=(
                    ResNet50_Weights.IMAGENET1K_V2
                )
            )


            # Remove:
            #
            # AdaptiveAvgPool
            # FC classifier

            backbone = nn.Sequential(

                *list(model.children())[:-2]

            )


            output_channels = 2048


        # ----------------------------------------------------
        # RESNET152
        # ----------------------------------------------------

        elif backbone_name == "resnet152":

            model = resnet152(

                weights=(
                    ResNet152_Weights.IMAGENET1K_V2
                )
            )


            backbone = nn.Sequential(

                *list(model.children())[:-2]

            )


            output_channels = 2048


        # ----------------------------------------------------
        # EFFICIENTNET-B3
        # ----------------------------------------------------

        elif backbone_name == "efficientnet_b3":

            model = efficientnet_b3(

                weights=(
                    EfficientNet_B3_Weights.IMAGENET1K_V1
                )
            )


            backbone = model.features


            output_channels = 1536


        return (
            backbone,
            output_channels,
        )


    # ========================================================
    # CNN → SEQUENCE CONVERSION
    # ========================================================

    def cnn_to_sequence(
        self,
        feature_map,
    ):

        # Expected CNN output:
        #
        # [B, C, H, W]
        #
        # Convert to:
        #
        # [B, H, W, C]

        sequence = feature_map.permute(
            0,
            2,
            3,
            1,
        )


        # Flatten spatial dimensions:
        #
        # [B, H, W, C]
        #
        # →
        #
        # [B, H*W, C]

        sequence = sequence.reshape(

            sequence.size(0),

            -1,

            sequence.size(-1),
        )


        return sequence


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        return_features=False,
        return_debug=False,
    ):


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        cnn_features = self.cnn_backbone(
            x
        )


        # ----------------------------------------------------
        # CNN → LSTM sequence
        # ----------------------------------------------------

        sequence = self.cnn_to_sequence(
            cnn_features
        )


        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        lstm_output, _ = self.lstm(
            sequence
        )


        # Last sequence position

        x = lstm_output[
            :,
            -1,
            :
        ]


        # ----------------------------------------------------
        # POST-LSTM
        # ----------------------------------------------------

        x = self.batch_norm(
            x
        )


        x = self.dropout(
            x
        )


        x = self.dense(
            x
        )


        deep_features = self.relu(
            x
        )


        # ----------------------------------------------------
        # DEBUG MODE
        # ----------------------------------------------------

        if return_debug:

            return {

                "cnn_feature_map":
                    cnn_features,

                "lstm_sequence":
                    sequence,

                "lstm_output":
                    lstm_output,

                "deep_features":
                    deep_features,
            }


        # ----------------------------------------------------
        # FEATURE EXTRACTION MODE
        # ----------------------------------------------------

        if return_features:

            return deep_features


        # ----------------------------------------------------
        # TEMPORARY NN CLASSIFICATION
        # ----------------------------------------------------

        logits = self.classifier(
            deep_features
        )


        return logits