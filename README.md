# ECG_Test_Resnet_Transformer

Here I want to test 3 approaches to [classification of ECG](https://physionet.org/content/challenge-2017/1.0.0/) into 3 classes: normal rhythm, arrhythmia, other diseases.

---

## BasicCNN (284,947 parameters)

Results for basic CNN architecture + logreg ontop of NN probabilities:
|               | Rand Init 1 | Rand Init 2 | Rand Init 3 | logreg on 9 values |
| ------------- | ----------- | ----------- | ----------- | ------------------ |
| Test F1 score | 0.741       | 0.785       | 0.765       | 0.757              |

---

## ResNetLikeCNNx3 (344,259 parameters)

Results for ResNet-like CNN architecture #1 + logreg ontop of NN probabilities:
|               | Rand Init 1 | Rand Init 2 | Rand Init 3 | logreg on 9 values |
| ------------- | ----------- | ----------- | ----------- | ------------------ |
| Test F1 score | 0.715       | 0.770       | 0.775       | 0.787              |

---

## ResNetLikeCNNx4 (309,051 parameters)

Results for ResNet-like CNN architecture #2 + logreg ontop of NN probabilities:
|               | Rand Init 1 | Rand Init 2 | Rand Init 3 | logreg on 9 values |
| ------------- | ----------- | ----------- | ----------- | ------------------ |
| Test F1 score | 0.740       | 0.777       | 0.750       | 0.761              |

---

## SmallerResNet4x2 (147,811 parameters)

Results for smaller ResNet-like CNN architecture #3 + logreg ontop of NN probabilities:
|               | Rand Init 1 | Rand Init 2 | Rand Init 3 | logreg on 9 values |
| ------------- | ----------- | ----------- | ----------- | ------------------ |
| Test F1 score | 0.797       | 0.802       | 0.795       | 0.782              |

---

## SmallerResNet4x3 (151,451 parameters)

Results for smaller ResNet-like CNN architecture #4 + logreg ontop of NN probabilities:
|               | Rand Init 1 | Rand Init 2 | Rand Init 3 | logreg on 9 values |
| ------------- | ----------- | ----------- | ----------- | ------------------ |
| Test F1 score | 0.796       | 0.782       | 0.790       | 0.802              |

---

Results for ResNet-like CNN architecture # 5,6 + logreg ontop of NN probabilities:

todo

---

Results for Transformer-like architecture + logreg ontop of NN probabilities:

todo