from rewarder import SeekerAgent, LLamaSeekerAgent, LLamaSeekerArguments, load_llama_seeker
import torch
import argparse
seed=42
torch.manual_seed(seed=seed)
torch.cuda.manual_seed_all(seed=seed)

seeker = load_llama_seeker()

contents = [
    
    {'content': "Hi, are you having a good day at the moment?", 'speaker': 'supporter'}, 
    {'content': "Today is okay, I guess. I' m just stressed.", 'speaker': 'seeker'}, 
    {'content': 'Would you desire to talk to me more about it?', 'speaker': 'supporter'},
            
        ]

output = seeker.response(contents)
print(output)