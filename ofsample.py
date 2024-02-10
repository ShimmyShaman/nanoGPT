"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from torch.nn import functional as F
import tiktoken
from model import GPTConfig, GPT
from odin_infill_encoder import OdinInfillEncoder, ENDOFTEXT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, ENDOFPROMPT, FILENAME

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 128 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    print(f'{gptconf.block_size=}')
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# # look for the meta pickle in case it is available in the dataset folder
# load_meta = False
# if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
#     meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
#     load_meta = os.path.exists(meta_path)
# if load_meta:
#     print(f"Loading meta from {meta_path}...")
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     # TODO want to make this more general to arbitrary encoder/decoder schemes
#     stoi, itos = meta['stoi'], meta['itos']
#     encode = lambda s: [stoi[c] for c in s]
#     decode = lambda l: ''.join([itos[i] for i in l])
# else:
#     # ok let's assume gpt-2 encodings by default
#     print("No meta.pkl found, assuming GPT-2 encodings...")
#     enc = tiktoken.get_encoding("gpt2")
#     encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
#     decode = lambda l: enc.decode(l)

def create_encoder():
    # mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
    #     vocab_bpe_file="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
    #     encoder_json_file="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    # )
    # for k, v in mergeable_ranks.items():
    #     print(f'{k=}  {v=}')

    mr=dict()
    for i in range(128):
        # Convert the integer to a character
        char = chr(i)
        # Convert the character to bytes using UTF-8 encoding
        byte_value = char.encode('utf-16')
        mr[byte_value] = i
    
    # for k, v in mr.items():
    #     print(f'{k=}  {v=}')
    # keywords = r"""asm|auto_cast bit_set break case cast context continue defer distinct do dynamic else enum fallthrough for foreign if import in map not_in or_else or_return package proc return struct switch transmute typeid union using when where
    return OdinInfillEncoder(
        name='coen',
        # explicit_n_vocab=134,
        vocabsize=134,
        # "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        pattern=r"",
        # pat_str=r"",
        mergeable_ranks=mr,
        special_tokens={ENDOFTEXT: 128, FIM_PREFIX: 129, FIM_MIDDLE: 130, FIM_SUFFIX: 131, ENDOFPROMPT: 132, FILENAME: 133})
encoder = create_encoder()

import pandas as pd
import numpy as np
import random

block_size = 512

numpy_array_prompts = np.array([])
numpy_array_expects = np.array([])
def load_data():
    df = pd.read_parquet('/media/rolly/Madrid/clc/odin_fillin1.parquet')

    padded_prompts = []
    padded_expects = []
    # Pad or truncate each sequence
    # [seq[:max_length] + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in df['prompt']
    ii = 0
    for pr in df['prompt']:
        if len(pr) > block_size:
            print(ii, f' load_data() prompt;   {len(pr)=} > {block_size=}')
            print(pr)
            exit(9)
            
        a = []
        a.extend(pr)
        for i in range(block_size - len(pr)):
            a.append(0)
        padded_prompts.append(a)
        ii += 1

    ii = 0
    for pr in df['expect']:
        if len(pr) > block_size:
            print(ii, f' load_data() expect;   {len(pr)=} > {block_size=}')
            print(pr)
            exit(9)

        a = []
        a.extend(pr)
        for i in range(block_size - len(pr)):
            a.append(0)
        padded_expects.append(a)
        ii += 1

    # Convert to a numpy array
    numpy_array_prompts = np.array(padded_prompts)
    numpy_array_expects = np.array(padded_expects)
    print(f'loaded {len(numpy_array_prompts)=} {len(numpy_array_expects)=}')
    return numpy_array_prompts, numpy_array_expects
numpy_array_prompts, numpy_array_expects = load_data()


# # encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
# start_ids = encode(start)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(1):
            ri = random.randint(0, len(numpy_array_prompts))
            x = torch.from_numpy((numpy_array_prompts[ri]).astype(np.int64)).unsqueeze(0)
            print(f'{x}')
            x = x.to('cuda')

            print(f'{x.device=}')
            # y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            y, _ = model.forward(x)
            y[0, 0, 0] = 0.0
            print(y)
            print(y.shape)
            print('---------------')
            # pluck the logits at the final step and scale by desired temperature
            y = y[:, -1, :] / temperature
            print(y)
            print(y.shape)
            top_k = 1
            if top_k is not None:
                v, vi = torch.topk(y, min(top_k, y.size(-1)))
                print(f'{min(top_k, y.size(-1))=}')
                print(f'{v=}')
                print(f'{vi=}')
                y[y < v[:, [-1]]] = -float('Inf')

            print('-----#####------')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(y, dim=-1)
            print(f'{probs=}')
            p, pi = torch.topk(y, min(top_k, y.size(-1)))
            print(f'{p=}')
            print(f'{pi=}')
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            print(f'{idx_next=}')
            # append sampled index to the running sequence and continue
            x = torch.cat((x, idx_next), dim=1)
            print(f'{x=}')
