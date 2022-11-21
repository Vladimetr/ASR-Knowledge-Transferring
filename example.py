import torch
from kt_module import from_yaml

train_module = from_yaml('config.yaml')

# Encoder outputs (from Wav2Vec)
# B - batch size
# L - max seq len
# H - encoder out dim
B, L, H = 3, 10, 768
encoder_outputs = torch.rand(B, L, H, dtype=torch.float32, 
                                      requires_grad=True)
mask = torch.ones(B, L, dtype=torch.bool)  # 0 value means invalid

target_sentences = [
    'привет как дела',
    'что нового',
    'давай пока'
]
assert B == len(target_sentences)

losses = train_module(encoder_outputs, mask, target_sentences)
# loss = ctc_loss + [weighted sum of these losses]
# (see eq.4 and eq.7 in paper)
print(losses)

# loss.backward()
