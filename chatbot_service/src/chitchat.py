import sys
import os
src_root = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(src_root, 'config.json')
chatbot_root = os.path.join(src_root, '..', '..', 'transfer-learning-conv-ai')
sys.path.append(chatbot_root)

import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import json

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, config, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(config['max_length']):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=config['device']).unsqueeze(0)
        token_type_ids = torch.tensor(instance['token_type_ids'], device=config['device']).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / config['temperature']
        logits = top_filtering(logits, top_k=config['top_k'], top_p=config['top_p'])
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if config['no_sample'] else torch.multinomial(probs, 1)
        if i < config['min_length'] and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def send_message(history, message, model_config):
    config = model_config['config']
    dataset = model_config['dataset']
    tokenizer = model_config['tokenizer']
    model = model_config['model']

    personalities = [dialog['personality'] for dataset in dataset.values() for dialog in dataset]
    if not history['personality']:
        history['personality'] = random.choice(personalities)

    history['messages'].append(tokenizer.encode(message))
    with torch.no_grad():
        out_ids = sample_sequence(history['personality'], history['messages'],
            tokenizer, model, config)

    history['messages'].append(out_ids)
    history['messages'] = history['messages'][-(2*config['max_history'] + 1):]
    response_msg = tokenizer.decode(out_ids, skip_special_tokens=True)

    return response_msg


def load_model_config():
    with open(config_path) as f:
        config = json.load(f)

    if config['seed'] != 0:
    	random.seed(config['seed'])
    	torch.random.manual_seed(config['seed'])
    	torch.cuda.manual_seed(config['seed'])

    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if config['model'] == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(config['model_checkpoint'])

    model = model_class.from_pretrained(config['model_checkpoint'])
    model.to(config['device'])
    add_special_tokens_(model, tokenizer)

    dataset = get_dataset(tokenizer, config['dataset_path'], config['dataset_cache'])

    return {
        'config': config,
        'model': model,
        'tokenizer': tokenizer,
        'dataset': dataset,
    }
