# ECG_Test_Resnet_Transformer

Here I want to test 3 approaches to [classification of ECG](https://physionet.org/content/challenge-2017/1.0.0/) into 3 classes: normal rhythm, arrhythmia, other diseases.

---

Results for basic CNN architecture + logreg ontop of NN probabilities:
|          | Rand Init 1 | Rand Init 2 | Rand Init 3 | logreg on 9 values |
| -------- | ----------- | ----------- | ----------- | ------------------ |
| F1 score | 0.741       | 0.785       | 0.765       | 0.757              |

---

Results for ResNet-like CNN architecture + logreg ontop of NN probabilities:

todo

---

Results for Transformer-like architecture + logreg ontop of NN probabilities:

todo