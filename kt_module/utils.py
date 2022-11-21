import torch.nn as nn
from abc import ABCMeta, abstractmethod
from . import lm as language_model


class KTModule(nn.Module, metaclass=ABCMeta):
    """
    Knowledge Transferring module for train ASR
    https://arxiv.org/abs/2203.03582
    """
    def __init__(self, encoder_dim, lm:dict, *args, **kwargs):
        """
        in_dim (int): encoder output dim H
        lm (dict): config for LM (see config.yaml)
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.lm = language_model.from_config(lm)
        self.lm_fdim = self.lm.get_fdim()           # E

    def forward(self, encoder_outputs, encoder_mask, target_sentences):
        """
        Get losses for given batch
        B - batch size
        L - max len of elements
        H - encoder output dim
        E - output dim
        Args:
            encoder_outputs (B, L, H)
            encoder_mask (B, L) int or bool: 0 is invalid
            target_sentences (list[str]): B-list of sentences
        Returns:
            dict with losses:
                "cosine": scalar
                other losses (optionally)
        """
        assert encoder_outputs.shape[ :2] == encoder_mask.shape[ :2], \
                "wrong sizes of 'encoder_outputs' or/and 'encoder_mask'"
        assert encoder_outputs.shape[0] == len(target_sentences), \
                "number of target sentences must be equal to B"


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
            encoder_mask (B, L) int or bool: padding mask of encoder outputs
                                      0 - is invalid items
            target_lengths (B) int: length of targets
        Return:
            tuple
              embeds (B, T, E) float32: for comparision with LM embeds
              mask (B, T) int: corresponding to target lens
              losses (dict): may be empty
        """
        pass
