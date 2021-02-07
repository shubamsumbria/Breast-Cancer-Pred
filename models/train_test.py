def train_n_test():
    from src import feat, models
    mdl, time, pred, prob=models.lr(feat.feature())
