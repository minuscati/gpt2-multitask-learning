'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`
'''
import os
import argparse
import random
import torch
import math
import json

import numpy as np
import torch.nn.functional as F

import sacrebleu
import re

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.args = args

    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    gpt_outputs = self.gpt(input_ids, attention_mask)
    hidden_states = gpt_outputs['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.15, max_length=128):
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    whitelist_chars = ['\n', ',', '.', ';', '?', '!', ':', '-', "'", '"']
    whitelist_ids = set()
    for char in whitelist_chars:
      whitelist_ids.update(self.tokenizer.encode(char))
      whitelist_ids.update(self.tokenizer.encode(' ' + char))

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature

      if repetition_penalty > 1.0:
        unique_tokens = torch.unique(token_ids[0])
        for token in unique_tokens:
          if token.item() in whitelist_ids:
              continue
          val = logits_last_token[0, token]
          logits_last_token[0, token] = val / repetition_penalty if val > 0 else val * repetition_penalty

      if top_k > 0:
        top_k_values, _ = torch.topk(logits_last_token, top_k)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits_last_token = torch.where(
          logits_last_token < min_top_k,
          torch.tensor(-float('Inf')).to(logits_last_token.device),
          logits_last_token
        )

      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
      top_p_mask[..., 0] = True
      filtered_probs = sorted_probs * top_p_mask
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def load_dev_alignment_data(file_path):
  """Parse validation set, extracting first 3 lines as Prompt and remaining 11 lines as Reference."""
  with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
  sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]
  prompts, references = [], []
  for sonnet in sonnets:
    lines = [line for line in sonnet.strip().split('\n') if line.strip()]
    if len(lines) >= 3:
      prompts.append('\n'.join(lines[:3]) + '\n')
      references.append('\n'.join(lines[3:]))
  return prompts, references


def train_and_get_best_state(args):
  """Executes the training loop and returns the optimal weights stored in memory."""
  if args.use_gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
  else:
    device = torch.device('cpu')

  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sonnet_dataset.collate_fn)
  dev_dataset = SonnetsDataset('data/TRUE_sonnets_held_out_dev.txt')
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

  model = SonnetGPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr)

  best_loss = float('inf')
  patience = 3
  patience_counter = 0
  best_model_state = None
  best_epoch = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss, num_batches = 0, 0
    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=True):
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids, b_mask = b_ids.to(device), b_mask.to(device)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      num_batches += 1

    # Validation phase
    model.eval()
    val_loss, num_val_batches = 0, 0
    with torch.no_grad():
      for batch in dev_dataloader:
        b_ids, b_mask = batch['token_ids'], batch['attention_mask']
        b_ids, b_mask = b_ids.to(device), b_mask.to(device)
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        val_loss += loss.item()
        num_val_batches += 1
    val_loss = val_loss / num_val_batches
    
    if val_loss < best_loss:
      best_loss = val_loss
      best_epoch = epoch
      patience_counter = 0
      best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
      patience_counter += 1
    if patience_counter >= patience:
      break

  return best_model_state, best_loss, best_epoch


def run_hyperparameter_search(base_args):
  """
  Optimized evaluation grid: Clustered by training setups.
  Trains each base setup once, then tests multiple decoding parameters via inference.
  """
  import math
  import gc
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
  
  # Define 4 distinct base training configurations
  training_setups = [
    {"lr": 5e-5, "batch_size": 8, "name": "Base_Setup_LR5e-5_BS8"},
    {"lr": 1e-5, "batch_size": 4, "name": "Base_Setup_LR1e-5_BS4"},
    {"lr": 3e-5, "batch_size": 4, "name": "Base_Setup_LR3e-5_BS4"},
    {"lr": 3e-5, "batch_size": 8, "name": "Base_Setup_LR3e-5_BS8"},
  ]
  
  # Map the 10 experimental configurations to their corresponding base training setup
  test_matrix = [
    # Decoding ablation variants using Base Setup 0 (5e-5, BS8)
    {"run_name": "Run_1_No_Penalty",     "setup_idx": 0, "top_k": 50, "top_p": 0.9,  "repetition_penalty": 1.0},
    {"run_name": "Run_2_Mild_Penalty",   "setup_idx": 0, "top_k": 50, "top_p": 0.9,  "repetition_penalty": 1.1},
    {"run_name": "Run_3_Std_Penalty",    "setup_idx": 0, "top_k": 50, "top_p": 0.9,  "repetition_penalty": 1.15},
    {"run_name": "Run_4_High_Penalty",   "setup_idx": 0, "top_k": 50, "top_p": 0.9,  "repetition_penalty": 1.3},
    {"run_name": "Run_5_Conservative_P", "setup_idx": 0, "top_k": 50, "top_p": 0.75, "repetition_penalty": 1.15},
    {"run_name": "Run_6_Creative_P",     "setup_idx": 0, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.15},
    {"run_name": "Run_7_Extreme_P",      "setup_idx": 0, "top_k": 50, "top_p": 0.99, "repetition_penalty": 1.15},
    
    # Independent training configurations
    {"run_name": "Run_8_LowLR_SmallBS",  "setup_idx": 1, "top_k": 50, "top_p": 0.9,  "repetition_penalty": 1.15},
    {"run_name": "Run_9_MidLR_SmallBS",  "setup_idx": 2, "top_k": 50, "top_p": 0.9,  "repetition_penalty": 1.15},
    {"run_name": "Run_10_MidLR_LargeBS", "setup_idx": 3, "top_k": 50, "top_p": 0.9,  "repetition_penalty": 1.15},
  ]

  # Step 1: Pre-train and store the 4 base model weights in memory
  pretrained_weights = {}
  for idx, setup in enumerate(training_setups):
    print(f"\n========== Training Base Setup {idx+1}/4 ({setup['name']}) ==========")
    seed_everything(base_args.seed)
    base_args.lr = setup["lr"]
    base_args.batch_size = setup["batch_size"]
    
    best_state, best_loss, best_epoch = train_and_get_best_state(base_args)
    pretrained_weights[idx] = {
        "state": best_state, "loss": best_loss, "epoch": best_epoch, "lr": setup["lr"], "bs": setup["batch_size"]
    }
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

  # Step 2: Loop through the 10 configurations and generate metrics using cached weights
  print("\n" + "="*50 + "\nBase fine-tuning completed. Launching automated multi-configuration evaluation...\n" + "="*50)
  dev_prompts, dev_references = load_dev_alignment_data("data/TRUE_sonnets_held_out_dev.txt")
  
  for run in test_matrix:
    print(f"Evaluating generation alignment for: {run['run_name']}...")
    setup_data = pretrained_weights[run["setup_idx"]]
    
    # Initialize a temporary model and load trained base weights
    base_args.lr = setup_data["lr"]
    base_args.batch_size = setup_data["bs"]
    model = SonnetGPT(base_args).to(device)
    model.load_state_dict({k: v.to(device) for k, v in setup_data["state"].items()})
    model.eval()
    
    # Inference only: Generate text using the unique decoding configuration
    generated_hypotheses = []
    for prompt in dev_prompts:
      encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
      _, output = model.generate(
          encoding['input_ids'], temperature=base_args.temperature,
          top_k=run["top_k"], top_p=run["top_p"], repetition_penalty=run["repetition_penalty"]
      )
      generated_hypotheses.append(output.strip())
      
    # Calculate chrF score
    chrf_score = sacrebleu.corpus_chrf(generated_hypotheses, [dev_references]).score
    
    # Write results to log file
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/log_{run['run_name']}.txt"
    with open(log_filename, 'w', encoding='utf-8') as f:
      f.write(f"Run Name: {run['run_name']}\n")
      f.write(f"Training Config: LR={setup_data['lr']} | BS={setup_data['bs']} | Best Epoch={setup_data['epoch']}\n")
      f.write(f"Decoding Config: Top-K={run['top_k']} | Top-P={run['top_p']} | RepPenalty={run['repetition_penalty']}\n")
      f.write(f"Validation Loss: {setup_data['loss']:.4f} | Perplexity: {math.exp(setup_data['loss']):.4f}\n")
      f.write(f"Final Validation corpus-chrF Score: {chrf_score:.2f}\n\n")
      f.write("="*40 + "\nGENERATED POEMS SAMPLE\n" + "="*40 + "\n")
      # Log the first two sonnets as validation samples
      for p, h, r in zip(dev_prompts[:2], generated_hypotheses[:2], dev_references[:2]):
        f.write(f"\nPrompt:\n{p}\nGenerated:\n{h}\nReference:\n{r}\n"+"-\n")
        
    print(f"   ---> Completed! Log saved to {log_filename} | chrF: {chrf_score:.2f}")
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

  print("\nAll evaluation matrix runs completed. Logs saved in logs/ directory.")


def get_args():
  """Parses command line arguments for training and generation."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=15)
  parser.add_argument("--use_gpu", action='store_true')

  # Decoding default configuration parameters
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--top_k", type=int, default=50)
  parser.add_argument("--top_p", type=float, default=0.9)
  parser.add_argument("--repetition_penalty", type=float, default=1.15)

  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=5e-5)
  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return add_arguments(args)


def add_arguments(args):
  """Appends deterministic architecture parameters mapped to model dimensions."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  return args


if __name__ == "__main__":
  args = get_args()
  run_hyperparameter_search(args)