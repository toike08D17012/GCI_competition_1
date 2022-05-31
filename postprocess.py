import pandas as pd

columns = ['TP', 'FP', 'TN', 'FN', 'recall', 'precision', 'accuracy', 'f1_score', ]


def evaluate(predicts, targets, model_name):
    TP, TN, FP, FN = 0, 0, 0, 0
    for predict, target in zip(predicts, targets):
        target = int(target)
        if target == 1:
            if predict == 1:
                TP += 1
            elif predict == 0:
                FN += 1
        elif target == 0:
            if predict == 1:
                FP += 1
            elif predict == 0:
                TN += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_score = 2 * recall * precision / (recall + precision)

    df = pd.DataFrame(
            [[
                TP,
                FP,
                TN,
                FN,
                recall,
                precision,
                accuracy,
                f1_score
            ]],
            columns=columns,
            index=[model_name]
        )

    return df