import torch
from transformers import BlenderbotSmallForConditionalGeneration
per_gpu_train_batch_size = 12
device = torch.device("cuda")
model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M", cache_dir='blender-small')
model = model.to(device)
while True:
    model.train()
    input = torch.ones((per_gpu_train_batch_size, 500), dtype=torch.long)
    input = input.to(device)
    out = model(input, decoder_input_ids=input)
    model.zero_grad()
    print("\rprocess is running...", end='')
