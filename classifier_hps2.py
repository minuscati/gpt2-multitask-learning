#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier on SST and CFIMDB
'''

import random, numpy as np, argparse
import itertools
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class GPT2SentimentClassifier(torch.nn.Module):
  '''
  This module performs sentiment classification using GPT2 in a cloze-style (fill-in-the-blank) task.

  In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
  Thus, your forward() should return one logit for each of the 5 classes.
  '''

  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()

    # Backward-compatible mode flag; detailed control is handled by
    # unfreeze_last_n + train_embeddings when provided.
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    num_gpt_layers = len(self.gpt.gpt_layers)

    # Resolve how many top transformer blocks to unfreeze.
    # None => preserve old behavior from fine_tune_mode.
    unfreeze_last_n = getattr(config, 'unfreeze_last_n', None)
    if unfreeze_last_n is None:
      unfreeze_last_n = 0 if config.fine_tune_mode == 'last-linear-layer' else num_gpt_layers
    elif unfreeze_last_n < 0:
      # -1 means "all transformer layers".
      unfreeze_last_n = num_gpt_layers
    unfreeze_last_n = max(0, min(unfreeze_last_n, num_gpt_layers))

    # Resolve embedding behavior.
    train_embeddings = getattr(config, 'train_embeddings', None)
    if train_embeddings is None:
      train_embeddings = (config.fine_tune_mode == 'full-model')

    # 1) Freeze all GPT params by default.
    for param in self.gpt.parameters():
      param.requires_grad = False

    # 2) Unfreeze selected top transformer layers.
    if unfreeze_last_n > 0:
      for layer in self.gpt.gpt_layers[-unfreeze_last_n:]:
        for param in layer.parameters():
          param.requires_grad = True
      # The final LN sits on top of transformer stack and should follow
      # transformer fine-tuning.
      for param in self.gpt.final_layer_norm.parameters():
        param.requires_grad = True

    # 3) Optionally unfreeze token/position embeddings.
    self.gpt.word_embedding.weight.requires_grad = bool(train_embeddings)
    self.gpt.pos_embedding.weight.requires_grad = bool(train_embeddings)

    ### TODO: Create any instance variables you need to classify the sentiment of BERT embeddings.
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)


  def forward(self, input_ids, attention_mask):
    '''Takes a batch of sentences and returns logits for sentiment classes'''

    ### TODO: The final GPT contextualized embedding is the hidden state of the last token.
    ###       HINT: You should consider what is an appropriate return value given that
    ###       the training loop currently uses F.cross_entropy as the loss function.
    outputs = self.gpt(input_ids, attention_mask)
    last_token = outputs['last_token']
    x = self.dropout(last_token)
    logits = self.classifier(x)
    return logits
    



class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


class SentimentTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    sent_ids = [x[1] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
  num_labels = {}
  data = []
  if flag == 'test':
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    b_labels = b_labels.flatten()
    y_true.extend(b_labels)
    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  acc = accuracy_score(y_true, y_pred)

  return acc, y_pred, y_true, sents, sent_ids


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
  output_dir = os.path.dirname(filepath)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def fmt_float(v):
  return f"{float(v):.8g}"


def trial_key(dataset_name, lr, dropout, fine_tune_mode, seed, unfreeze_last_n, train_embeddings, weight_decay):
  return (
    f"{dataset_name}|{fine_tune_mode}|seed{seed}|lr{fmt_float(lr)}|drop{fmt_float(dropout)}|"
    f"unf{unfreeze_last_n}|emb{int(bool(train_embeddings))}|wd{fmt_float(weight_decay)}"
  )


def load_logged_trials(log_path):
  completed = set()
  entries = []
  if not os.path.exists(log_path):
    return completed, entries

  with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        item = json.loads(line)
      except json.JSONDecodeError:
        continue
      entries.append(item)
      if item.get('status') == 'done' and item.get('trial_key'):
        completed.add(item['trial_key'])
  return completed, entries


def append_search_log(log_path, payload):
  with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(payload, ensure_ascii=True) + '\n')


def append_epoch_log(log_path, payload):
  if not log_path:
    return
  output_dir = os.path.dirname(log_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(payload, ensure_ascii=True) + '\n')


def make_trial_checkpoint_path(args, dataset_name, lr, dropout):
  ckpt_dir = os.path.join(args.search_output_dir, 'checkpoints')
  unfreeze_tag = "all" if args.unfreeze_last_n < 0 else str(args.unfreeze_last_n)
  emb_tag = int(bool(args.train_embeddings))
  filename = (
    f"{dataset_name}_{args.fine_tune_mode}_seed{args.seed}_"
    f"lr{fmt_float(lr)}_drop{fmt_float(dropout)}_"
    f"unf{unfreeze_tag}_emb{emb_tag}_wd{fmt_float(args.weight_decay)}.pt"
  )
  return os.path.join(ckpt_dir, filename)


def train(args, trial_meta=None, epoch_log_path=None):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if epoch_log_path is None:
    epoch_log_path = getattr(args, 'epoch_log_path', None)
  # Create the data and its corresponding datasets and dataloader.
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

  # Init model.
  config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            'num_labels': num_labels,
            'hidden_size': 768,
            'data_dir': '.',
            'fine_tune_mode': args.fine_tune_mode,
            'unfreeze_last_n': args.unfreeze_last_n,
            'train_embeddings': args.train_embeddings}

  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
  best_dev_acc = 0
  best_epoch = -1
  epochs_without_improve = 0
  epoch_metrics = []

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / (num_batches)

    train_acc, *_ = model_eval(train_dataloader, model, device)
    dev_acc, *_ = model_eval(dev_dataloader, model, device)

    if dev_acc > best_dev_acc + args.early_stop_min_delta:
      best_dev_acc = dev_acc
      best_epoch = epoch
      epochs_without_improve = 0
      save_model(model, optimizer, args, config, args.filepath)
    else:
      epochs_without_improve += 1

    epoch_entry = {
      'timestamp': datetime.now(timezone.utc).isoformat(),
      'epoch': epoch,
      'train_loss': train_loss,
      'train_acc': train_acc,
      'dev_acc': dev_acc,
      'best_dev_acc_so_far': best_dev_acc,
      'best_epoch_so_far': best_epoch
    }
    if trial_meta:
      epoch_entry.update(trial_meta)
    epoch_metrics.append(epoch_entry)
    append_epoch_log(epoch_log_path, epoch_entry)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    if args.early_stop_patience >= 0 and epochs_without_improve >= args.early_stop_patience:
      print(f"Early stopping triggered at epoch {epoch} "
            f"(best_dev_acc={best_dev_acc:.4f} at epoch {best_epoch})")
      break

  stopped_epoch = epoch_metrics[-1]['epoch'] if epoch_metrics else -1
  return {
    'best_dev_acc': best_dev_acc,
    'best_epoch': best_epoch,
    'stopped_epoch': stopped_epoch,
    'epoch_metrics': epoch_metrics
  }


def build_dataset_config(dataset_name, args, lr=None, dropout=None):
  lr = args.lr if lr is None else lr
  dropout = args.hidden_dropout_prob if dropout is None else dropout
  unfreeze_tag = "all" if args.unfreeze_last_n is not None and args.unfreeze_last_n < 0 else str(args.unfreeze_last_n)
  emb_tag = str(args.train_embeddings)
  run_tag = (
    f"{dataset_name}_{args.fine_tune_mode}_seed{args.seed}_"
    f"lr{fmt_float(lr)}_drop{fmt_float(dropout)}_"
    f"unf{unfreeze_tag}_emb{emb_tag}_wd{fmt_float(args.weight_decay)}"
  )
  epoch_log_path = os.path.join(args.search_output_dir, f'epoch_metrics_{run_tag}.jsonl')

  if dataset_name == 'sst':
    return SimpleNamespace(
      filepath='hyperresults/ptfiles/sst-classifier.pt',
      lr=lr,
      use_gpu=args.use_gpu,
      epochs=args.epochs,
      batch_size=args.batch_size,
      hidden_dropout_prob=dropout,
      weight_decay=args.weight_decay,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=args.fine_tune_mode,
      unfreeze_last_n=args.unfreeze_last_n,
      train_embeddings=args.train_embeddings,
      early_stop_patience=args.early_stop_patience,
      early_stop_min_delta=args.early_stop_min_delta,
      epoch_log_path=epoch_log_path,
      dev_out='predictions/' + args.fine_tune_mode + '-sst-dev-out.csv',
      test_out='predictions/' + args.fine_tune_mode + '-sst-test-out.csv'
    )
  elif dataset_name == 'cfimdb':
    return SimpleNamespace(
      filepath='hyperresults/ptfiles/cfimdb-classifier.pt',
      lr=lr,
      use_gpu=args.use_gpu,
      epochs=args.epochs,
      batch_size=8,
      hidden_dropout_prob=dropout,
      weight_decay=args.weight_decay,
      train='data/ids-cfimdb-train.csv',
      dev='data/ids-cfimdb-dev.csv',
      test='data/ids-cfimdb-test-student.csv',
      fine_tune_mode=args.fine_tune_mode,
      unfreeze_last_n=args.unfreeze_last_n,
      train_embeddings=args.train_embeddings,
      early_stop_patience=args.early_stop_patience,
      early_stop_min_delta=args.early_stop_min_delta,
      epoch_log_path=epoch_log_path,
      dev_out='predictions/' + args.fine_tune_mode + '-cfimdb-dev-out.csv',
      test_out='predictions/' + args.fine_tune_mode + '-cfimdb-test-out.csv'
    )

  raise ValueError(f'Unknown dataset: {dataset_name}')


def run_lr_dropout_search(args):
  lrs = [float(x) for x in args.search_lrs.split(',')]
  dropouts = [float(x) for x in args.search_dropouts.split(',')]
  datasets = ['sst', 'cfimdb'] if args.search_dataset == 'both' else [args.search_dataset]

  os.makedirs(args.search_output_dir, exist_ok=True)
  log_path = os.path.join(args.search_output_dir, f'lr_dropout_trials_seed{args.seed}_{args.fine_tune_mode}.jsonl')
  epoch_log_path = os.path.join(args.search_output_dir, f'epoch_metrics_seed{args.seed}_{args.fine_tune_mode}.jsonl')
  completed_keys, historical_entries = load_logged_trials(log_path)
  results = [x for x in historical_entries if x.get('status') == 'done']
  if completed_keys:
    print(f"Loaded {len(completed_keys)} completed trial(s) from {log_path}")

  for dataset_name in datasets:
    print(f"\n=== Running lr/dropout search on {dataset_name} ===")
    best_result = None
    for lr, dropout in itertools.product(lrs, dropouts):
      key = trial_key(
        dataset_name, lr, dropout, args.fine_tune_mode, args.seed,
        args.unfreeze_last_n, args.train_embeddings, args.weight_decay
      )
      if key in completed_keys:
        print(f"\n[skip done] {key}")
        continue

      print(f"\n[search] dataset={dataset_name}, lr={lr}, dropout={dropout}")
      seed_everything(args.seed)
      run_config = build_dataset_config(dataset_name, args, lr=lr, dropout=dropout)
      run_config.filepath = make_trial_checkpoint_path(args, dataset_name, lr, dropout)

      try:
        run_metrics = train(
          run_config,
          trial_meta={
            'trial_key': key,
            'dataset': dataset_name,
            'fine_tune_mode': args.fine_tune_mode,
            'seed': args.seed,
            'lr': lr,
            'dropout': dropout,
            'unfreeze_last_n': args.unfreeze_last_n,
            'train_embeddings': bool(args.train_embeddings),
            'weight_decay': args.weight_decay
          },
          epoch_log_path=epoch_log_path
        )
        entry = {
          'timestamp': datetime.now(timezone.utc).isoformat(),
          'status': 'done',
          'trial_key': key,
          'dataset': dataset_name,
          'fine_tune_mode': args.fine_tune_mode,
          'seed': args.seed,
          'lr': lr,
          'dropout': dropout,
          'unfreeze_last_n': args.unfreeze_last_n,
          'train_embeddings': bool(args.train_embeddings),
          'weight_decay': args.weight_decay,
          'checkpoint_path': run_config.filepath,
          'best_dev_acc': run_metrics['best_dev_acc'],
          'best_epoch': run_metrics['best_epoch'],
          'stopped_epoch': run_metrics['stopped_epoch']
        }
        append_search_log(log_path, entry)
        results.append(entry)
        completed_keys.add(key)
        print(f"[trial done] dataset={dataset_name}, lr={lr}, dropout={dropout}, "
              f"best_dev_acc={entry['best_dev_acc']:.4f}")
        if best_result is None or entry['best_dev_acc'] > best_result['best_dev_acc']:
          best_result = entry
      except Exception as e:
        fail_entry = {
          'timestamp': datetime.now(timezone.utc).isoformat(),
          'status': 'failed',
          'trial_key': key,
          'dataset': dataset_name,
          'fine_tune_mode': args.fine_tune_mode,
          'seed': args.seed,
          'lr': lr,
          'dropout': dropout,
          'unfreeze_last_n': args.unfreeze_last_n,
          'train_embeddings': bool(args.train_embeddings),
          'weight_decay': args.weight_decay,
          'checkpoint_path': run_config.filepath,
          'error': str(e)
        }
        append_search_log(log_path, fail_entry)
        print(f"[failed] {key} -> {e}")
        continue

    dataset_done = [x for x in results if x.get('dataset') == dataset_name and x.get('status') == 'done']
    if dataset_done:
      best_result = max(dataset_done, key=lambda x: x['best_dev_acc'])
    if best_result is None:
      print(f"[best] {dataset_name}: no successful trial yet.")
    else:
      print(f"[best] {dataset_name}: lr={best_result['lr']}, dropout={best_result['dropout']}, "
            f"dev_acc={best_result['best_dev_acc']:.4f}")

  timestamp = f"seed{args.seed}_{args.fine_tune_mode}"
  output_path = os.path.join(args.search_output_dir, f'lr_dropout_search_{timestamp}.json')
  with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
  print(f"\nSaved search results to {output_path}")
  print(f"Trial log (resume source): {log_path}")


def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    dev_acc, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print('DONE DEV')

    test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print('DONE Test')

    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.0)
  parser.add_argument("--weight_decay", type=float, default=0.0,
                      help="Weight decay used by AdamW.")
  parser.add_argument("--unfreeze_last_n", type=int, default=None,
                      help="How many top transformer layers to unfreeze. "
                           "0=linear probe, positive int=partial FT, -1=all.")
  parser.add_argument("--train_embeddings", type=int, choices=(0, 1), default=None,
                      help="Whether to train token/position embeddings: 0 freeze, 1 train. "
                           "If omitted, follows fine_tune_mode behavior.")
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-3)
  parser.add_argument("--early_stop_patience", type=int, default=2,
                      help="Unified early stopping patience for all runs. "
                           "Set -1 to disable early stopping.")
  parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                      help="Minimum dev_acc improvement required to reset early stopping.")
  parser.add_argument("--run_hpsearch", action='store_true',
                      help="Run lr x dropout hyper-parameter search instead of default train/test flow.")
  # parser.add_argument("--search_lrs", type=str, default="5e-4,1e-3,2e-3,5e-3,1e-2",
  parser.add_argument("--search_lrs", type=str, default="5e-4",
                      help="Comma-separated learning rates for search.")
  # parser.add_argument("--search_dropouts", type=str, default="0.0,0.2",
  parser.add_argument("--search_dropouts", type=str, default="0.005",
                      help="Comma-separated dropout rates for search.")
  parser.add_argument("--search_dataset", type=str, default="sst", choices=("sst", "cfimdb", "both"),
                      help="Which dataset(s) to run hyper-parameter search on.")
  parser.add_argument("--search_output_dir", type=str, default="hpsearch_results",
                      help="Directory to save hyper-parameter search results.")

  args = parser.parse_args()
  if args.train_embeddings is not None:
    args.train_embeddings = bool(args.train_embeddings)
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)

  if args.run_hpsearch:
    run_lr_dropout_search(args)
    raise SystemExit(0)

  print('Training Sentiment Classifier on SST...')
  config = build_dataset_config('sst', args)

  train(config)

  print('Evaluating on SST...')
  test(config)

  print('Training Sentiment Classifier on cfimdb...')
  config = build_dataset_config('cfimdb', args)

  train(config)

  print('Evaluating on cfimdb...')
  test(config)
