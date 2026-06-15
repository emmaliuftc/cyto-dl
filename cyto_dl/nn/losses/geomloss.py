"""
Adapted from: https://github.com/jeanfeydy/geomloss
LICENSE: https://github.com/jeanfeydy/geomloss/blob/main/LICENSE.txt
"""

import torch
import torch.nn as nn
from geomloss import SamplesLoss


class GeomLoss(nn.Module):
    def __init__(
        self,
        name: str = "sinkhorn",
        p: int = 1,
        blur: float = 0.01,
        reach=None,
        scaling=0.5,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.p = p
        self.blur = blur
        self.reach = reach
        self.scaling = scaling
        kwargs.pop("_aux", None)

        self.loss = SamplesLoss(
            loss=self.name,
            p=self.p,
            blur=self.blur,
            reach=self.reach,
            scaling=self.scaling,
            **kwargs,
        )

    def forward(self, preds, gts):
        # 1. FORCE CHANNELS-LAST FORMAT
        # If the last dimension is larger than the second dimension (e.g., 2048 points vs 4 channels),
        # it means the tensor is channels-first. We transpose it to [Batch, Points, Channels].
        if len(preds.shape) == 3 and preds.shape[-1] > preds.shape[1]:
            preds = preds.transpose(1, 2)

        if len(gts.shape) == 3 and gts.shape[-1] > gts.shape[1]:
            gts = gts.transpose(1, 2)

        # 2. STRIP DUMMY FEATURES
        # GeomLoss only calculates spatial distance in 3D.
        # If the model outputted 4 channels, we slice off everything except the first 3 (X, Y, Z).
        if preds.shape[-1] > 3:
            preds = preds[..., :3]

        if gts.shape[-1] > 3:
            gts = gts[..., :3]

        return self.loss(preds, gts)
