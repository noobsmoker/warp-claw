#!/usr/bin/env python3
"""
Model Downloader for Warp-Claw
Downloads HuggingFace models with M1 optimization.
"""

import os
import sys
import argparse
from pathlib import Path


def get_model_config(model_id: str) -> dict:
    """Get model configuration."""
    import yaml
    
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        return config.get("models", {}).get(model_id, {})


def download_model(
    model_id: str,
    tokenizer_only: bool = False,
    quant: str = "q4_k"
):
    """Download model from HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    config = get_model_config(model_id)
    repo = config.get("repo", f"Qwen/{model_id}")
    
    print(f"📥 Downloading {model_id} from {repo}")
    
    # Create cache directory
    cache_dir = Path(__file__).parent.parent / "data" / "models" / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download tokenizer
    print("  📝 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        repo,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    tokenizer.save_pretrained(cache_dir)
    
    if tokenizer_only:
        print("  ✅ Tokenizer only - skipping model")
        return
    
    # Download model with MPS/CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  📥 Downloading model to {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        device_map=device,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    model.save_pretrained(cache_dir)
    
    print(f"  ✅ Model saved to {cache_dir}")
    
    # Print size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    print(f"  📊 Total size: {total_size / 1024 / 1024:.1f} MB")


def list_available_models():
    """List available models."""
    import yaml
    
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    
    if not config_path.exists():
        print("No models.yaml found")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        models = config.get("models", {})
    
    print("Available models:")
    for model_id, cfg in models.items():
        print(f"  {model_id}: {cfg.get('repo')}")
        print(f"    Max agents: {cfg.get('max_agents')}")


def main():
    parser = argparse.ArgumentParser(description="Download models for Warp-Claw")
    parser.add_argument("model", nargs="?", help="Model ID to download")
    parser.add_argument("--tokenizer-only", action="store_true", help="Download tokenizer only")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--quant", default="q4_k", help="Quantization (q4_k, q5_k, etc)")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.model:
        parser.print_help()
        return
    
    download_model(args.model, args.tokenizer_only, args.quant)


if __name__ == "__main__":
    main()