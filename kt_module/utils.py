import torch.nn as nn
from abc import ABCMeta, abstractmethod


class RLmechanism(nn.Module, metaclass=ABCMeta):
    """
    Reprepresentation Learning mechanism
    fig. 1 in paper 
    """
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        """
        in_dim (int): encoder out feature dim H
        out_dim (int): output embed dim E
        """
        super().__init__()
        self.train()  # only for train mode

    def eval(self):
        raise NotImplementedError("RL mechanism is only for train")

    @abstractmethod
    def forward(self, encoder_outputs, encoder_mask, target_lengths):
        """
        H - encoder embed dim
        L - encoder max len
        T - target max len
        E - out embeds dim (equal to LM dim)
        Args:
            encoder_outputs (B, L, H) float32: raw outputs 
                                               of acoustic encoder
            mask (B, L) bool: padding mask of encoder outputs
                              0 - is invalid items
            target_lengths (B) int: length of targets
        Return:
            tuple
              embeds (B, T, E) float32: for comparision with LM embeds
              mask (B, T) int: corresponding to target lens
              losses (dict): may be empty
        """
        pass
