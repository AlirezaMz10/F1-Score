# F1 Score Calculation in Python

## Overview
This project implements a simple Python script to compute the F1 score, a key performance metric for classification models. The script calculates True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) from a given dataset and derives accuracy, precision, recall, and the F1 score.

## Formulae
The script uses the following formulae to compute the evaluation metrics:

- **Accuracy**:  
  \[ \frac{TP + TN}{TP + FP + TN + FN} \]

- **Precision**:  
  \[ \frac{TP}{TP + FP} \]

- **Recall**:  
  \[ \frac{TP}{TP + FN} \]

- **F1 Score**:  
  \[ 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]


## Implementation
The script is implemented using `numpy` for efficient array operations.

```python
import numpy as np

df = np.array([
    [True, False],
    [True, False],
    [True, True],
    [False, True],
    [False, False],
    [True, False],
    [True, True],
    [False, True],
    [False, False],
    [True, False],
    [True, False],
    [True, False],
    [True, True],
    [True, True],
    [False, True],
])

positives = df[df[:, 1] == True]
tp = positives[positives[:, 0] == True]
tp_count = tp.shape[0]

fp = positives[positives[:, 0] == False]
fp_count = fp.shape[0]

negatives = df[df[:, 1] == False]
tn = negatives[negatives[:, 0] == False]
tn_count = tn.shape[0]

fn = negatives[negatives[:, 0] == True]
fn_count = fn.shape[0]

accuracy = (tp_count + tn_count) / (tp_count + fp_count + tn_count + fn_count)
precision = round(tp_count / (tp_count + fp_count), 3)
recall = round(tp_count / (tp_count + fn_count), 3)
f1_score = round(2 * precision * recall / (precision + recall), 3)

print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1_score:", f1_score)
```

## Output Example
After running the script, the output will display the values for accuracy, precision, recall, and F1 score.

Example output:
```
count of TP: 4
count of FP: 3
count of TN: 2
count of FN: 6
accuracy: 0.4
precision: 0.571
recall: 0.4
f1_score: 0.47
```

## Requirements
Make sure you have Python installed along with `numpy`. You can install `numpy` using:
```
pip install numpy
```

## License
This project is open-source and available under the MIT License.

