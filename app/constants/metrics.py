from tensorflow.keras import metrics


class Metrics:

    def __init__(self):
        self.TRUE_POSITIVE = metrics.TruePositives(name='tp')
        self.FALSE_POSITIVE = metrics.FalsePositives(name='fp')
        self.TRUE_NEGATIVE = metrics.TrueNegatives(name='tn')
        self.FALSE_NEGATIVE = metrics.FalseNegatives(name='fn')
        self.BINARY_ACCURACY = metrics.BinaryAccuracy(name='accuracy')
        self.PRECISION = metrics.Precision(name='precision')
        self.RECALL = metrics.Recall(name='recall')
        self.AUC = metrics.AUC(name='auc')
