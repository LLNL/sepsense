import torch

class Pass(torch.nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x

class Reshape(torch.nn.Module):
    def __init__(self, num_classes):
        super(Reshape, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x.view(x.size(0), self.num_classes)

class Reshape2(torch.nn.Module):
    def __init__(self):
        super(Reshape2, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), 512, 13, 13)
