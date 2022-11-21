import torch
import torch.nn as nn
from .utils import KTModule


class RepresentationLearning(KTModule):
    """
    KT - Representation Learning (RL) module
    """
    def __init__(self, 
                encoder_dim,        # encoder output dim H
                lm:dict,            # config (see config.yaml)
                mechanism:dict,     # config (see config.yaml)
        ):
        super().__init__(encoder_dim=encoder_dim, lm=lm)
        self._init_mechanism(self.encoder_dim, self.lm_fdim, mechanism)
        # Init Losses
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        # Only train mode is available for RL
        self.train()

    def _init_mechanism(self, in_dim, out_dim, mechanism_cfg:dict):
        name = mechanism_cfg.pop("name")
        if name == 'cif':
            from .cif import CIF
            self.mechanism = CIF(
                in_dim=in_dim,
                out_dim=out_dim,
                **mechanism_cfg
            )
        elif name == 'attention':
            from .attention import Attention
            self.mechanism = Attention(
                in_dim=in_dim,
                out_dim=out_dim,
                **mechanism_cfg
            )
        self.mechanism.train()  # only train mode is available

    def eval(self):
        raise NotImplementedError("Only train mode is available for RL")

    def forward(self, encoder_outputs, encoder_mask, target_sentences):
        """
        Get losses for given batch
        B - batch size
        L - max len of elements
        H - encoder output dim
        E - CIF out dim (equal to LM feature dim)
        Args:
            encoder_outputs (B, L, H)
            encoder_mask (B, L) int or bool: 0 is invalid
            target_sentences (list[str]): B-list of sentences
        Returns:
            dict with losses:
                "cosine": scalar
                other losses (optionally)
        """    
        # check sizes
        super().forward(encoder_outputs, encoder_mask, target_sentences)

        # LM is not trainable
        with torch.no_grad():
            lm_embeds, target_mask = self.lm.get_embeds(target_sentences)
        # (B, T, E), (B, T) int

        # mechanism
        target_lengths = target_mask.sum(-1)
        output = self.mechanism(
            encoder_outputs=encoder_outputs,
            encoder_mask=encoder_mask,
            target_lengths=target_lengths,
        )
        out_embeds, out_mask, losses = output

        # check embeds match
        assert lm_embeds.shape == out_embeds.shape, \
                    f"{lm_embeds.shape} {out_embeds.shape}"
        # check mask match
        assert torch.equal(target_mask.bool(), out_mask.bool())

        # mask apply
        mask = torch.unsqueeze(out_mask, -1).int()
        out_embeds *= mask
        lm_embeds *= mask
        # (B, T, E)
        
        # Loss computation
        cos_loss = 1 - self.cos_sim(out_embeds, lm_embeds)
        # take into account only unmasked values
        cos_loss *= out_mask
        # mean
        cos_loss = torch.sum(cos_loss) / torch.sum(out_mask)
        # scalar
        losses["cosine"] = cos_loss.item()
        
        return losses
