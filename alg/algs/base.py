import torch


class Algorithm(torch.nn.Module):
    def __init__(self, args):
        super(Algorithm, self).__init__()

    def update(self, minibatches):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
