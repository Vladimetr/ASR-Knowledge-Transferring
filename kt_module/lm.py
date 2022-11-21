import torch
from abc import ABCMeta, abstractmethod
from transformers import AutoTokenizer
from transformers import AutoModel


def from_config(config:dict):
    """
    Args:
        config (dict): see config.yaml:lm
    Returns:
        Bert, 
    """
    config = dict(config)  # copy
    name = config.pop("name")
    if name == 'bert':
        lm = Bert
    elif name == 'gpt2':
        lm = GPT2
    else:
        raise ValueError(f"Invalid LM '{name}'")
    return lm(**config)


class LanguageModel(metaclass=ABCMeta):
    def __init__(self, tokenizer:str, model:str):
        """
        Load Tokenizer and LM Model from huggingface hub
        https://huggingface.co/models
        tokenizer (str): from huggingface hub
        model (str): from huggingface hub
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModel.from_pretrained(model)
        self.config = self.model.config.to_dict()
        # set PAD token to tokenizer as it specified in model
        pad_id = self.config.get("pad_token_id", None)
        if pad_id is None:
            pad_id = self.config["eos_token_id"]
        self.tokenizer.pad_token_id = pad_id
        # check vocab size
        if self.tokenizer.vocab_size > self.config["vocab_size"]:
            raise ValueError("Tokenizer vocab is bigger than model vocab")

    @staticmethod
    def _preprocess(text):
        text = text.lower()
        # tokenizer will do other operations 
        return text 

    def get_fdim(self):
        """
        Get last hidden feature dim of LM embeds
        Returns:
            int
        """
        return self.config["hidden_size"]

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def tokenize(self, texts):
        """
        Tokenize sentences
        Args:
            texts (list[str]): B-list of sentences
            NOTE: one batch element is only one sentence!
        Returns:
            tuple
              tensor (B, L) int: ids
              tensor (B, L) int: mask (0 is masked)
        """
        texts = [self._preprocess(text) for text in texts]
        tokens = self.tokenizer(texts, 
                                padding=True, 
                                # as sentences have different lens
                                add_special_tokens=False,
                                # no need of SOS, SEP, EOS
                                return_tensors="pt"
                                )
        ids = tokens["input_ids"]
        mask = tokens["attention_mask"]
        return ids, mask

    @abstractmethod
    def get_embeds(self, texts, return_ids=False):
        """
        Get embeds for batch of sentences
        B - number of sentences (batch size)
        L - max length of sentence in the batch
        F - embed size
        Args:
            texts (list[str]): B-list of sentences
                NOTE: one batch element is only one sentence!
            return_tokens (bool): whether to return list of tokens
        Returns:
            tuple
              tensor (B, L, F) float32: embeds
              tensor (B, L) int: mask (0 is masked)
              tensor (B, L) int: token IDs (if return_ids=True)
        """
        pass


class Bert(LanguageModel):
    def __init__(self, tokenizer:str, model:str):
        """
        Load Tokenizer and BertModel from huggingface hub
        tokenizer (str): for ex. 'sberbank-ai/ruBert-base'
        model (str): for ex. 'sberbank-ai/ruBert-base'
        """
        super().__init__(tokenizer, model)

    def get_embeds(self, texts, return_ids=False):
        """
        See description in parent class
        """
        token_ids, mask = self.tokenize(texts)
        # (B, L)
        outputs = self.model(input_ids=token_ids, 
                             attention_mask=mask)
        embeds = outputs.last_hidden_state
        # (B, L, F)

        if return_ids:
            return embeds, mask, token_ids
        return embeds, mask


class GPT2(LanguageModel):
    def __init__(self, tokenizer:str, model:str):
        """
        Load Tokenizer and GPT2Model from huggingface hub
        tokenizer (str): for ex. 'sberbank-ai/rugpt2large'
        model (str): for ex. 'sberbank-ai/rugpt2large'
        """
        super().__init__(tokenizer, model)

    def get_fdim(self):
        try:
            fdim = self.config["hidden_size"]
        except KeyError:
            fdim = self.config["n_embd"]
        return fdim

    def get_embeds(self, texts, return_ids=False):
        """
        See description in parent class
        """
        token_ids, mask = self.tokenize(texts)
        # (B, L)
        outputs = self.model(input_ids=token_ids, 
                             attention_mask=mask)
        embeds = outputs.last_hidden_state
        # (B, L, F)

        # NOTE that GPT2 is designed to predict next token,
        # so out embeds are shifted right along L axis
        # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt2/modeling_gpt2.py#L1072
        # For example: [1, 2, 3, 4] -> [2, 3, 4, 5]
        # where 5 - is predicted future token
        # according to prev tokens 1, 2, 3, 4

        # that's why we need to shift back
        start_pos = torch.zeros_like(embeds[:, :1, :])  # (B, 1, F)
        embeds = torch.cat((start_pos, embeds[:, :-1, :]), dim=1)
        # (B, L, F)

        if return_ids:
            return embeds, mask, token_ids
        return embeds, mask



if __name__ == '__main__':
    lm = Bert('sberbank-ai/ruBert-base', 'sberbank-ai/ruBert-base')
    texts = [
        'привет как дела',
        'что нового',
        'давай пока'
    ]
    ids, _ = lm.tokenize(texts)
    print(ids)
    # (B, L)

    embeds, mask = lm.get_embeds(texts)
    print(mask)
    # (B, L, F), (B, L)
