# PHASE 2A / 2B — FIXED HORIZON, HYPERPARAMETER SENSITIVITY, AND FULL-HYBRID STATE EVALUATION

## Phase2A question
What does each backbone do under the same 30-epoch horizon when early stopping is removed as a confounder?

## Phase2A design
Baseline configuration:
- RMSprop
- LR1e-4
- LSTM256
- dropout0.3
- Dense128
- batch8
- seed42
- exactly 30 epochs
- retain best-validation checkpoint and epoch30 state

## Phase2A training observations
Neural peak/final behavior came from `phase2_training_40run_summary.csv`. The project used full training curves to separate peak accuracy from stability.

Representative full-hybrid state selected per backbone:
| Backbone | State | Epoch | SVM val acc | Macro F1 | Internal test acc* |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b3 | best_val_within_30 | 27 | 99.300% | 99.270% | 99.314% |
| resnet152 | best_val_within_30 | 10 | 98.600% | 98.545% | 98.703% |
| resnet50 | epoch_30 | 30 | 98.688% | 98.644% | 98.932% |
| vgg16 | epoch_30 | 30 | 95.888% | 95.746% | 96.186% |
| vgg19 | epoch_30 | 30 | 95.626% | 95.462% | 95.271% |

Important: for VGG16/VGG19/ResNet50, best and epoch30 downstream performance tied or the epoch30 state represented the same endpoint; ResNet152 and EfficientNet-B3 retained earlier best states.

## Phase2B question
Which of three tested factors — learning rate, LSTM hidden size, dropout — most changes performance under an otherwise fixed protocol?

## Phase2B one-factor configurations
| Run | LR | LSTM | Dropout |
| --- | ---: | ---: | ---: |
| lr_1e-5 | 1e-5 | 256 | 0.3 |
| lr_5e-5 | 5e-5 | 256 | 0.3 |
| lr_5e-4 | 5e-4 | 256 | 0.3 |
| lstm_128 | 1e-4 | 128 | 0.3 |
| lstm_512 | 1e-4 | 512 | 0.3 |
| dropout_0.2 | 1e-4 | 256 | 0.2 |
| dropout_0.5 | 1e-4 | 256 | 0.5 |

5 backbones × 7 = 35 changed-config training runs.

## Full execution counts
- Phase2A training: 5/5
- Phase2B training: 35/35
- combined fixed-30 training histories: 40
- Phase2A retained-state hybrid evaluations: 10/10
- Phase2B retained-state hybrid evaluations: 70/70
- total Phase2 downstream state evaluations in the package: 80

## Phase2B strongest changed-config full-hybrid states
| Backbone | Selected changed config | State | Epoch | SVM val acc | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b3 | lstm_512 | best_val_within_30 | 29 | 99.388% | 99.377% |
| resnet152 | lstm_512 | best_val_within_30 | 26 | 99.300% | 99.265% |
| resnet50 | dropout_0.2 | best_val_within_30 | 28 | 99.125% | 99.122% |
| vgg16 | lr_5e-4 | best_val_within_30 | 27 | 96.500% | 96.321% |
| vgg19 | dropout_0.5 | epoch_30 | 30 | 95.538% | 95.382% |

## Sensitivity magnitude
Mean peak-neural-validation within-backbone ranges:
- learning rate: 4.62 pp
- LSTM width: 0.58 pp
- dropout: 0.42 pp

### Learning-rate behavior
- VGG16/VGG19: LR1e-5 severely underfit; LR5e-4 tolerated relatively well.
- ResNet50/ResNet152: LR5e-4 was damaging and unstable.
- EfficientNet-B3: baseline 1e-4 strongest among LR-only settings.

Conclusion: **learning-rate preference is architecture-dependent**.

### LSTM width
LSTM512 raised peak neural validation versus LSTM256 in all five, but only modestly. Some peaks deteriorated by epoch30.

Conclusion: **modest peak improvement association**, not universal superiority.

### Dropout
Dropout changes had smaller and inconsistent effects.

## Late-epoch stability
Strongest example:
ResNet50 LR5e-4:
- best neural validation ≈94.75% at epoch25
- epoch30 ≈90.99%
- drift ≈-3.76 pp

This is why the project retained fixed-horizon curves.

## Downstream-state insight
Across the 35 changed configs:
- best-neural state higher SVM val: 20
- epoch30 higher: 8
- identical: 7

This proves that “best neural checkpoint” and “best downstream SVM state” are not always synonymous.

## Limits
- one seed;
- one known-contaminated saved validation split;
- no statistical significance testing;
- only three hyperparameter families;
- one-factor design does not test interactions.

## Evidence files
- `results/phase2_training_40run_summary.csv`
- `results/phase2_full_hybrid_80_state_evaluations.csv`
- `results/phase2_selected_states_by_backbone.csv`
- `raw_packages/phase2_full_analysis_package.zip`
