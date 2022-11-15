import math
import torch
import torch.nn as nn
from .utils import RLmechanism


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, fdim, max_len=5000, dropout=0.1):
        """
        fdim (int): input feature dim H
        max_len (int): L
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (L, 1)
        d = torch.arange(0, fdim, 2) * (-math.log(10000.0) / fdim)  # (L, )
        div_term = torch.exp(d)  # (L, )
        pe = torch.zeros(1, max_len, fdim)  # (B=1, L, H)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        B - batch size
        L - max len
        H - embed dim
        Args:
            x (B, L, H)
        Returns:
            (B, L, H)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Attention(RLmechanism):
    def __init__(self, in_dim, out_dim, nheads=4, 
                       dropout=0.1, bias=True):
        """
        in_dim (int): encoder output dim H
        out_dim (int): mechanism output dim E
        nhead (int): number of attention heads
        """
        super().__init__(in_dim, out_dim)
        self.pos_encoder = PositionalEncoding(in_dim, dropout=dropout)
        self.attention = nn.MultiheadAttention(
                            embed_dim=in_dim,
                            num_heads=nheads,
                            dropout=dropout,
                            bias=bias
        )
        self.encoder_embed_dim = in_dim
        # for matching last dims
        if in_dim != out_dim:
            self.out_layer = nn.Linear(in_dim, 
                                       out_dim, 
                                       bias=bias)
        else:
            self.out_layer = None

    @staticmethod
    def _mask_from_lengths(lengths):
        """
        in: [2, 1, 4]
        out: [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1]
        ]
        Args:
            lengths (B) int
        Returns:
            (B, L) bool: 0 is masked. L - max length
        """
        n = len(lengths)
        max_len = lengths.max()
        target_mask = torch.zeros(n, max_len, dtype=torch.bool)
        for i in range(n):
            cur_len = lengths[i].item()
            target_mask[i, :cur_len] = 1.
        return target_mask

    def forward(self, encoder_outputs, encoder_mask, target_lengths):
        """
        see more description in parent class
        """
        target_mask = self._mask_from_lengths(target_lengths).int()
        target_mask_exp = torch.unsqueeze(target_mask, -1)  # expanded
        # (B, T, 1)
        empty_tensor = torch.zeros_like(target_mask_exp)
        pos_embeds = self.pos_encoder(empty_tensor)
        pos_embeds *= target_mask_exp
        # (B, T, H) with mask applied (may be wrong!)

        # NOTE: In Multihead attention batch dim is second.
        # In some PyTorch verions it's available to change it
        # but in some versions it's not
        # (B, T, ) -> (L, T, )
        pos_embeds = torch.transpose(pos_embeds, 0, 1)
        # (B, L, ) -> (L, B, )
        encoder_outputs = torch.transpose(encoder_outputs, 0, 1)

        # Q (T, B, H)
        # K, V (L, B, H)
        query = pos_embeds
        key = encoder_outputs
        value = encoder_outputs
        attn_output, _ = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=1-encoder_mask.int(),
            need_weights=False
        )
        # (T, B, H) -> (B, T, H)
        out = torch.transpose(attn_output, 0, 1)
        
        # (B, T, H) -> (B, T, E)
        if self.out_layer:
            out = self.out_layer(out)

        # add losses
        losses = dict()  # no losses provided in this mechanism

        return out, target_mask, losses



if __name__ == '__main__':
    att = Attention(512, 768)
    B, L = 3, 10
    H = 512
    encoder_outputs = torch.rand(B, L, H, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.int)

    target_lens = torch.tensor([2, 3, 4], dtype=torch.int)

    out = att(encoder_outputs, mask, target_lens)
    print(out.shape)

    