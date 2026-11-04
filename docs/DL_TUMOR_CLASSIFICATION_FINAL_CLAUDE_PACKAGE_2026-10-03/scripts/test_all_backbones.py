from pathlib import Path
import sys
import gc

import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model_cnn_lstm import (
    CNNLSTM,
    SUPPORTED_BACKBONES,
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


print("=" * 80)

print(
    "GENERIC CNN + LSTM BACKBONE TEST"
)

print("=" * 80)


print(
    "\nDevice:",
    device
)


if torch.cuda.is_available():

    print(

        "GPU:",

        torch.cuda.get_device_name(0)

    )


# ============================================================
# TEST CONFIGURATION
# ============================================================

BATCH_SIZE = 2


# Small batch because all five models are tested
# sequentially.

test_input = torch.randn(

    BATCH_SIZE,

    3,

    224,

    224,

    device=device,
)


# ============================================================
# TEST EACH BACKBONE
# ============================================================

for backbone_name in SUPPORTED_BACKBONES:


    print(
        "\n"
        + "=" * 80
    )


    print(
        f"TESTING: "
        f"{backbone_name.upper()}"
    )


    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = CNNLSTM(

        backbone_name=backbone_name,

        lstm_hidden_size=256,

        dense_features=128,

        num_classes=4,

        dropout=0.3,
    )


    model = model.to(
        device
    )


    model.eval()


    # --------------------------------------------------------
    # FORWARD DEBUG PASS
    # --------------------------------------------------------

    with torch.no_grad():

        debug = model(

            test_input,

            return_debug=True,
        )


        logits = model(
            test_input
        )


    # --------------------------------------------------------
    # OUTPUT SHAPES
    # --------------------------------------------------------

    cnn_shape = tuple(

        debug[
            "cnn_feature_map"
        ].shape

    )


    sequence_shape = tuple(

        debug[
            "lstm_sequence"
        ].shape

    )


    lstm_shape = tuple(

        debug[
            "lstm_output"
        ].shape

    )


    deep_shape = tuple(

        debug[
            "deep_features"
        ].shape

    )


    logits_shape = tuple(
        logits.shape
    )


    print(
        "\nCNN feature map :",
        cnn_shape
    )


    print(
        "LSTM sequence   :",
        sequence_shape
    )


    print(
        "LSTM output     :",
        lstm_shape
    )


    print(
        "Deep features   :",
        deep_shape
    )


    print(
        "Class logits    :",
        logits_shape
    )


    # --------------------------------------------------------
    # ASSERTIONS
    # --------------------------------------------------------

    assert (
        sequence_shape[0]
        == BATCH_SIZE
    )


    assert (
        sequence_shape[2]
        == model.cnn_output_channels
    )


    assert lstm_shape == (

        BATCH_SIZE,

        sequence_shape[1],

        256,
    )


    assert deep_shape == (

        BATCH_SIZE,

        128,
    )


    assert logits_shape == (

        BATCH_SIZE,

        4,
    )


    assert torch.isfinite(

        debug[
            "deep_features"
        ]

    ).all()


    print(
        "\n✓ BACKBONE TEST PASSED"
    )


    # --------------------------------------------------------
    # CLEAN GPU MEMORY BEFORE NEXT BACKBONE
    # --------------------------------------------------------

    del model

    del debug

    del logits


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 80
)


print(
    "ALL BACKBONES PASSED"
)


print(
    "=" * 80
)


print(
    "\nThe generic CNN + LSTM architecture "
    "is ready for comparative training."
)