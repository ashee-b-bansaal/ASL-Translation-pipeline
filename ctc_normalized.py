import torch
import torch.nn.functional as F
from models import Imu1dImage2dCTC


class Imu1dImage2dCTCNormalized(Imu1dImage2dCTC):
    """
    Wrapper over Imu1dImage2dCTC that normalizes the temporal dimension of logits
    to a fixed K using adaptive average pooling. No training hyperparameters change.
    """

    def __init__(self, *args, normalize_k: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_k = normalize_k

    def _normalize_logits_to_k(self, logits: torch.Tensor) -> torch.Tensor:
        """Resample (Tw,B,V+1) logits to (K,B,V+1) using adaptive average pooling."""
        Tw, B, Vp1 = logits.shape
        if Tw == self.normalize_k:
            return logits
        # (Tw,B,V+1) -> (B,V+1,Tw)
        logits_bvt = logits.permute(1, 2, 0)
        # Pool to K
        logits_bvk = F.adaptive_avg_pool1d(logits_bvt, self.normalize_k)
        # (B,V+1,K) -> (K,B,V+1)
        logits_kbv = logits_bvk.permute(2, 0, 1)
        return logits_kbv

    def forward(self, x_img: torch.Tensor, x_imu: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x_img, x_imu)  # (Tw,B,V+1)
        return self._normalize_logits_to_k(logits)


