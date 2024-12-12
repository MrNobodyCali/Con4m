from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class EvaluationIndex:
    def __init__(self, acc=0., pre=0., rec=0., f1=0.):
        # Index for classification
        self.acc = acc
        self.pre = pre
        self.rec = rec
        self.f1 = f1

    def __str__(self):
        out = ''
        out += '-' * 10 + 'Evaluation Result' + '-' * 10 + '\n'
        out += 'Accuracy: ' + str(self.acc) + '\n'
        out += 'Precision: ' + str(self.pre) + '\n'
        out += 'Recall: ' + str(self.rec) + '\n'
        out += 'F1: ' + str(self.f1) + '\n'
        return out


def class_evaluation(y_true, y_pred, n_class):
    accuracy = accuracy_score(y_true, y_pred)
    if n_class == 2:
        average = 'binary'
    else:
        average = 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
    )

    index = EvaluationIndex(accuracy, precision, recall, f1)
    return index
