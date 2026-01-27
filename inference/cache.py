import torch


class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        k_new, v_new: (B, H, 1, Dh)
        """
        if self.keys is None:
            self.keys = k_new
            self.values = v_new
        else:
            self.keys = torch.cat([self.keys, k_new], dim=2)
            self.values = torch.cat([self.values, v_new], dim=2)

    def reset(self):
        self.keys = None
        self.values = None
