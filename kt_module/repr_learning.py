import torch
import torch.nn as nn
from .lm import from_config

class RepresentationLearning(nn.Module):
    """
    Knowledge Transfering Representation Learning module
    https://arxiv.org/pdf/2203.03582.pdf
    """
    def __init__(self, 
                in_dim,             # encoder out feature dim H
                out_dim,            # out dim E
                lm:dict,            # config (see config.yaml:bert)
                mechanism:dict,     # config (see config.yaml:mechanism)
        ):
        super().__init__()
        self._init_mechanism(in_dim, out_dim, mechanism)
        self.lm = from_config(lm)
        if out_dim != self.lm.get_fdim():
            raise Exception("Mismatch feature dims")
        # Init Losses
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        # Only train mode is available
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
        raise NotImplementedError("Only train mode is available")

    def forward(self, encoder_outputs, encoder_mask, target_sentences):
        """
        Get losses for given batch
        B - batch size
        L - max len of elements
        H - encoder output dim
        E - CIF out dim (equal to LM feature dim)
        Args:
            encoder_outputs (B, L, H)
            encoder_mask (B, L) int: 0 is invalid
            target_sentences (list[str]): B-list of sentences
        Returns:
            dict with losses:
                "cosine": tensor (B, )
                    cosine similarity btwn 
                    CIF outputs and LM embeds 
                        (B, T): if reduction=None
                        scalar: if reduction='mean'
                other losses (optionally)

        """    
        # LM is not trainable
        with torch.no_grad():
            lm_embeds, target_mask = self.lm.get_embeds(target_sentences)
        # (B, T, E), (B, T)

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

        # Loss computation
        cos_loss = 1 - self.cos_sim(out_embeds, lm_embeds)
        # (B, )
        losses["cosine"] = cos_loss
        return losses
