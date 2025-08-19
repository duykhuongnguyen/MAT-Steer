#!/usr/bin/env python3
"""
Evaluation script for MAT-Steer checkpoints on TruthfulQA and other tasks.
"""

import argparse
import sys
import os
sys.path.append('..')

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import MAT-Steer components
from interveners import MATIntervener, create_mat_pyvene_config
import pyvene as pv

# Import evaluation utilities
from utils import alt_tqa_evaluate, ENGINE_MAP

def load_mat_checkpoint(checkpoint_path):
    """Load MAT-Steer checkpoint and return intervener."""
    return MATIntervener.load_from_checkpoint(checkpoint_path)

def evaluate_truthfulqa(model, tokenizer, mat_intervener, layer, instruction_prompt="default"):
    """Evaluate MAT-Steer model on TruthfulQA."""
    
    # Create pyvene config for the specified layer
    pv_config = create_mat_pyvene_config([layer], mat_intervener)
    
    # Wrap model with interventions
    intervenable_model = pv.IntervenableModel(pv_config, model)
    
    # Run TruthfulQA evaluation
    print(f"Evaluating on TruthfulQA with MAT intervention at layer {layer}...")
    results = alt_tqa_evaluate(
        intervenable_model, 
        tokenizer, 
        instruction_prompt=instruction_prompt,
        num_samples=None  # Use all samples
    )
    
    return results

def compute_baseline_metrics(model, tokenizer, instruction_prompt="default"):
    """Compute baseline metrics without intervention."""
    print("Computing baseline metrics...")
    results = alt_tqa_evaluate(
        model, 
        tokenizer, 
        instruction_prompt=instruction_prompt,
        num_samples=None
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate MAT-Steer checkpoint")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name (e.g., llama3.1_8B)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to MAT-Steer checkpoint (.pt file)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index for intervention (auto-detect from metadata if not specified)")
    parser.add_argument("--instruction_prompt", type=str, default="default",
                        help="Instruction prompt type for evaluation")
    parser.add_argument("--multiplier", type=float, default=1.0,
                        help="Intervention strength multiplier")
    parser.add_argument("--baseline", action="store_true",
                        help="Also compute baseline metrics without intervention")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    if args.model_name not in ENGINE_MAP:
        raise ValueError(f"Unknown model: {args.model_name}. Available: {list(ENGINE_MAP.keys())}")
    
    model_name_or_path = ENGINE_MAP[args.model_name]
    print(f"Loading model: {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Load MAT checkpoint
    print(f"Loading MAT checkpoint: {args.checkpoint}")
    mat_intervener = MATIntervener.load_from_checkpoint(
        args.checkpoint, 
        multiplier=args.multiplier
    )
    
    # Determine layer from metadata if not specified
    layer = args.layer
    if layer is None:
        metadata_path = args.checkpoint.replace('.pt', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                layer = metadata.get('layer', 14)
                print(f"Using layer {layer} from metadata")
        else:
            layer = 14  # Default
            print(f"No metadata found, using default layer {layer}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Baseline evaluation
    if args.baseline:
        baseline_results = compute_baseline_metrics(model, tokenizer, args.instruction_prompt)
        baseline_path = os.path.join(args.output_dir, f"{args.model_name}_baseline_results.json")
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print(f"Baseline results saved to: {baseline_path}")
    
    # MAT-Steer evaluation
    mat_results = evaluate_truthfulqa(model, tokenizer, mat_intervener, layer, args.instruction_prompt)
    
    # Save results
    results_filename = f"{args.model_name}_L{layer}_mat_results.json"
    results_path = os.path.join(args.output_dir, results_filename)
    
    # Add metadata to results
    mat_results['evaluation_metadata'] = {
        'model_name': args.model_name,
        'checkpoint_path': args.checkpoint,
        'layer': layer,
        'multiplier': args.multiplier,
        'instruction_prompt': args.instruction_prompt
    }
    
    with open(results_path, 'w') as f:
        json.dump(mat_results, f, indent=2)
    
    print(f"MAT-Steer evaluation results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if args.baseline and 'baseline_results' in locals():
        print("BASELINE METRICS:")
        for key, value in baseline_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        print()
    
    print("MAT-STEER METRICS:")
    for key, value in mat_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    print("="*60)

if __name__ == "__main__":
    main()
