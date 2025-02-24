# Pyvene method of getting activations
import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('../')

import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import load_dataset, get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_2, tokenized_tqa_gen, tokenized_tqa_gen_end_q
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama3.1_8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1_8B_chat': 'mathewhe/Llama-3.1-8B-Chat',
    'qwen2.5_7B': 'Qwen/Qwen2.5-7B',
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    dataset = load_dataset(args.dataset_name)
    if args.dataset_name == "truthfulqa": 
        formatter = tokenized_tqa
    elif args.dataset_name == "toxigen" or args.dataset_name == "bbq": 
        formatter = tokenized_tqa_2
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    prompts, labels = formatter(dataset, tokenizer)

    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = []
    token_labels = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
        for i in range(layer_wise_activations.shape[1]):
            all_layer_wise_activations.append(layer_wise_activations[args.layer,i,:].copy())
            token_labels.append(i)
        all_head_wise_activations.append(head_wise_activations.copy())

    print("Saving labels")
    np.save(f'../features/{args.model_name}_{args.dataset_name}_labels.npy', labels)
    
    print("Saving token labels")
    np.save(f'../features/{args.model_name}_{args.dataset_name}_token_labels.npy', token_labels)

    print("Saving layer wise activations")
    np.save(f'../features/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'../features/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()
