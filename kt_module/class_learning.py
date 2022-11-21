import torch
import torch.nn as nn
from .utils import KTModule


class ClassificationLearning(KTModule):
    """
    KT - Classification Learning (RL) module
    """
    def __init__(self, encoder_dim,     # encoder out dim H
                       lm:dict,         # see config.yaml
                       nheads=4,        # number of attention heads
                       dropout=0.1,
                       bias=True,
        ):
        super().__init__(encoder_dim=encoder_dim, lm=lm)
        self.n_tokens = self.lm.get_vocab_size()  # C
        self.nheads = nheads
        self.dim_matcher = None
        self.out_layer = None
        # for matching dims
        if self.lm_fdim != self.encoder_dim:
            # E -> H
            self.dim_matcher = nn.Linear(self.lm_fdim,       # E
                                         self.encoder_dim,   # H
                                         bias=bias)
        # cross attention
        self.attention = nn.MultiheadAttention(
                        embed_dim=self.encoder_dim,
                        num_heads=nheads,
                        dropout=dropout,
                        bias=bias
        )
        # token classification layer
        self.out_layer = nn.Linear(self.encoder_dim, 
                                   self.n_tokens,  # C
                                   bias=bias)
        ignore_index = self.lm.get_pad_token_id()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

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

    @staticmethod
    def _combine_masks(mask1, mask2):
        """
        Args:
            mask1 (B, S1) int or bool: [[1, 1, 0, 0]]
            mask2 (B, S2) int or bool: [[1, 1, 1, 0, 0, 0, 0]]
        Raises:
            ValueError: "mask1 and mask2 must have save type"
        Returns:
            (B, S1, S2) int or bool: logical AND
            [[[1, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]]
        """
        if mask1.dtype != mask2.dtype:
            raise ValueError("mask1 and mask2 must have save type")
        B, S1 = mask1.shape
        _, S2 = mask2.shape

        mask1 = torch.unsqueeze(mask1, 2)
        # (B, S1, 1)
        mask1 = mask1.expand(B, S1, S2)
        # (B, S1, S2)

        mask2 = torch.unsqueeze(mask2, 1)
        # (B, 1, S2)
        mask2 = mask2.expand(B, S1, S2)
        # (B, S1, S2)

        if mask1.dtype == torch.bool:
            combined_mask = torch.logical_and(mask1, mask2)
        else:
            combined_mask = mask1 * mask2

        return combined_mask

    def forward(self, encoder_outputs, encoder_mask, target_sentences):
        """
        Get losses for given batch
        B - batch size
        L - max len of elements
        H - encoder output dim
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

        B, L, _ = encoder_outputs.shape
        # LM is not trainable
        with torch.no_grad():
            lm_embeds, target_mask, token_ids = \
                 self.lm.get_embeds(target_sentences, 
                                    return_ids=True)
        # (B, T, E), (B, T) int, (B, T)
        T = target_mask.shape[1]

        # (B, T, E) -> (B, T, H)
        if self.dim_matcher:
            lm_embeds = self.dim_matcher(lm_embeds)

        # NOTE: In Multihead attention batch dim is second for Q, K, V.
        # In some PyTorch versions it's available to change it
        # but in some versions it's not
        # (B, T, *) -> (T, B, *)
        lm_embeds = torch.transpose(lm_embeds, 0, 1)
        encoder_outputs = torch.transpose(encoder_outputs, 0, 1)

        # combine input and output masks -> attn mask
        target_mask = target_mask.bool()
        encoder_mask = encoder_mask.bool()
        attn_mask = self._combine_masks(target_mask, encoder_mask)
        # (B, T, L)
        attn_mask = torch.unsqueeze(attn_mask, 1)  # (B, 1, T, L)
        attn_mask = attn_mask.expand(-1, self.nheads, -1, -1)
        attn_mask = attn_mask.reshape(B*self.nheads, T, L)
        # (B*HEADS, T, L) transformer requires this format
        
        # cross attention (like encoder-decoder attention in Transformer)
        attn_output, _ = self.attention(
            query=lm_embeds,
            key=encoder_outputs,
            value=encoder_outputs,
            key_padding_mask=torch.logical_not(encoder_mask),
            attn_mask=torch.logical_not(attn_mask),
            need_weights=False,
        )
        # (T, B, H) -> (B, T, H)
        attn_output = torch.transpose(attn_output, 0, 1)

        # classification layer
        token_logits = self.out_layer(attn_output)
        # (B, T, C)

        # check if masking is correct
        assert torch.equal(
            torch.logical_not(torch.isnan(token_logits[:, :, 0])),
            target_mask
        )

        # Loss computation
        token_logits = token_logits.view(-1, self.n_tokens)
        # (B*T, C)
        token_ids = token_ids.view(-1)
        # (B*T)
        ce_loss = self.ce_loss(token_logits, token_ids)
        assert not torch.isnan(ce_loss)
        losses = {
            "ce": ce_loss
        }
        return losses
