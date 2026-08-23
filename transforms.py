import torch

class ContrastiveTransform:

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2
        
    def __call__(self, x):
        view1 = self.transform1(x)
        view2 = self.transform2(x)
        return view1, view2

class ToCDHW:

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.permute(1, 0, 2, 3)
        return sample
