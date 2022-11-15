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
        err_message = 'GPT2 is not implemented yet'
        raise NotImplementedError(err_message)
    else:
        raise ValueError(f"Invalid LM '{name}'")
    return lm(**config)


class Bert:
    def __init__(self, tokenizer:str, model:str):
        """
        Load Tokenizer and BertModel from huggingface hub
        https://huggingface.co/models
        tokenizer (str): for ex. 'sberbank-ai/ruBert-base'
        model (str): for ex. 'sberbank-ai/ruBert-base'
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModel.from_pretrained(model)
        self.config = self.model.config.to_dict()

    @staticmethod
    def _preprocess(text):
        text = text.lower()
        # tokenizer will do other operations 
        return text

    def get_fdim(self):
        """
        Get last hidden feature dim of Bert
        Returns:
            int
        """
        return self.config["hidden_size"]

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

    def get_embeds(self, texts):
        """
        Get embeds for batch of sentences
        B - number of sentences (batch size)
        L - max length of sentence in the batch
        F - embed size
        Args:
            texts (list[str]): B-list of sentences
            NOTE: one batch element is only one sentence!
        Returns:
            tuple
              tensor (B, L, F) float32: embeds
              tensor (B, L) int: mask (0 is masked)
        """
        token_ids, mask = self.tokenize(texts)
        # (B, L)
        embeds = self.model(input_ids=token_ids, 
                            attention_mask=mask)
        return embeds.last_hidden_state, mask



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
