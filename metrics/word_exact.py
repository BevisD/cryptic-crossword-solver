from torchmetrics import Metric
import torch

__all__ = ['WordExact']


class WordExact(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: list[str], target: list[str]) -> None:
        if len(preds) != len(target):
            raise ValueError(f"Length of predictions and target do not match: ({len(preds)}), ({len(target)})")

        for pred, label in zip(preds, target):
            if pred == label:
                self.correct += 1
        self.total += len(preds)

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total
