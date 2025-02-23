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
print(df)
print(df.shape)

positives = df[df[:, 1] == True]
print(positives)

tp = positives[positives[:, 0] == True]
tp_count =  tp.shape[0]
print("count of TP:", tp_count)

fp = positives[positives[:, 0] == False]
fp_count =  fp.shape[0]
print("count of FP:", fp_count)

negatives = df[df[:, 1] == False]
print(negatives)

tn = negatives[negatives[:, 0] == False]
tn_count =  tn.shape[0]
print("count of TN:", tn_count)

fn = negatives[negatives[:, 0] == True]
fn_count =  fn.shape[0]
print("count of TN:", fn_count)

accuracy = (tp_count + tn_count) / (tp_count + fp_count + tn_count + fn_count)
print("accuracy:", accuracy)

precision = round(tp_count / (tp_count + fp_count), 3)
print("precision:", precision)

recall = round(tp_count / (tp_count + fn_count), 3)
print("recall:", recall)

f1_score = round(2 * precision * recall / (precision + recall), 3)
print("f1_score:", f1_score)