#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier on SST and CFIMDB
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

import os
import sys
from datetime import datetime
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

def setup_experiment(args):
    import os
    import sys
    import json
    from datetime import datetime

    # ===== 创建时间戳 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== 实验名字 =====
    if args.fine_tune_mode == "lora":
        exp_name = (
            f"{args.fine_tune_mode}"
            f"_r{args.lora_r}"
            f"_a{args.lora_alpha}"
            f"_d{args.lora_dropout}"
            f"_lr{args.lr}"
            f"_bs{args.batch_size}"
            f"_ep{args.epochs}"
            f"_{timestamp}"
        )
    else:
        exp_name = (
            f"{args.fine_tune_mode}"
            f"_lr{args.lr}"
            f"_bs{args.batch_size}"
            f"_ep{args.epochs}"
            f"_{timestamp}"
        )

    # ===== 创建实验目录 =====
    save_dir = os.path.join("experiments", exp_name)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    # ===== log 文件 =====
    log_path = os.path.join(save_dir, "train.log")

    # ===== Logger =====
    class Logger(object):
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log = open(log_file, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # ===== 保存 terminal 输出 =====
    sys.stdout = Logger(log_path)

    # ===== 打印实验信息 =====
    print("=" * 80)
    print("Experiment Created")
    print("=" * 80)

    print(f"Save Directory:\n{save_dir}\n")

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("=" * 80)

    # ===== 保存 config.json =====
    config_path = os.path.join(save_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    print(f"Config saved to:\n{config_path}")
    print(f"Log file saved to:\n{log_path}")

    print("=" * 80)

    return save_dir


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

    # # Pretrain mode does not require updating GPT paramters.
    # assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    # for param in self.gpt.parameters():
    #   if config.fine_tune_mode == 'last-linear-layer':
    #     param.requires_grad = False
    #   elif config.fine_tune_mode == 'full-model':
    #     param.requires_grad = True

    # Pretrain mode does not require updating GPT paramters.
    assert config.fine_tune_mode in ["last-linear-layer", "full-model", "lora"] # 加了 lora
    for param in self.gpt.parameters():
      if config.fine_tune_mode in ['last-linear-layer', 'lora']: # 加了 lora
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    ### TODO: Create any instance variables you need to classify the sentiment of BERT embeddings.
    ### YOUR CODE HERE


    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

    # raise NotImplementedError



  # def forward(self, input_ids, attention_mask):
  #   '''Takes a batch of sentences and returns logits for sentiment classes'''

  #   # 1. 跑一遍 GPT-2 拿到输出字典
  #   gpt_outputs = self.gpt(input_ids, attention_mask)
    
  #   # 2. 提取最后一个 token 的上下文表示 (表示整个句子)
  #   last_token_state = gpt_outputs['last_token']
    
  #   # 3. 依次通过 Dropout 和 线性分类层
  #   x = self.dropout(last_token_state)
  #   logits = self.classifier(x)
    
  #   # 返回未经 softmax 的 logits (因为训练循环里用了 F.cross_entropy)
  #   return logits


#   def forward(self, input_ids, attention_mask):

#       gpt_outputs = self.gpt(input_ids, attention_mask)

#       hidden_states = gpt_outputs['last_hidden_state']

#       seq_lengths = attention_mask.sum(dim=1) - 1

#       last_token_state = hidden_states[
#           torch.arange(hidden_states.size(0)),
#           seq_lengths
#       ]

#       x = self.dropout(last_token_state)

#       logits = self.classifier(x)

#       return logits
  
  # new forward for mean pooling
  def forward(self, input_ids, attention_mask):

    gpt_outputs = self.gpt(input_ids, attention_mask)

    hidden_states = gpt_outputs['last_hidden_state']

    # [B, T, H]
    mask = attention_mask.unsqueeze(-1)

    # padding 部分清零
    masked_hidden = hidden_states * mask

    # mean pooling
    sum_hidden = masked_hidden.sum(dim=1)

    lengths = mask.sum(dim=1)

    pooled = sum_hidden / lengths

    x = self.dropout(pooled)

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
    with open(filename, 'r',encoding='utf-8') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    with open(filename, 'r',encoding='utf-8') as fp:
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

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sents, sent_ids


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


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
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
            'fine_tune_mode': args.fine_tune_mode}

  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)

  # config = SimpleNamespace(**config)

  # model = GPT2SentimentClassifier(config)
  
  # === 新增：注入 LoRA 模块 ===
  if args.fine_tune_mode == 'lora':
    lora_config = LoraConfig(
        # r=16,
        # lora_alpha=32,
        # target_modules=["query", "value"], # 匹配你 attention.py 的命名
        # target_modules=["query", "key", "value", "attention_dense", "interm_dense", "out_dense"],
        target_modules=["query", "key", "value", "attention_dense", "interm_dense", "out_dense"],
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # lora_dropout=0.2,
        bias="none",
        modules_to_save=["classifier"] # ⚠️ 极其重要：把你的分类头加入训练
    )
    model = get_peft_model(model, lora_config)
    print("\n" + "=" * 80)
    print("LoRA Model Info")
    print("=" * 80)

    # PEFT 自带
    model.print_trainable_parameters()

    # ===== trainable parameter 数量 =====
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    ratio = 100 * trainable_params / total_params

    print(f"Trainable params : {trainable_params:,}")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable ratio  : {ratio:.4f}%")

    # ===== 保存 model info =====
    model_info_path = os.path.join(args.save_dir, "model_info.txt")

    with open(model_info_path, "w") as f:
        f.write(f"Trainable params: {trainable_params:,}\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Trainable ratio: {ratio:.4f}%\n")

    print(f"Model info saved to:\n{model_info_path}")

  model = model.to(device)

  lr = args.lr
  # === 修改：优化器只传入 requires_grad=True 的参数，避免无谓的显存占用和报错 ===
#   optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

  # 修改：添加 weight_decay 参数进行 L2 正则化，帮助防止过拟合（尤其是在全模型微调时）

  optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=0.01)

  # model = model.to(device)

  # lr = args.lr
  # optimizer = AdamW(model.parameters(), lr=lr)
  best_dev_acc = 0

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
    #   loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
      

      # === 修改：添加 label_smoothing 参数 ===
      loss = F.cross_entropy(logits,b_labels.view(-1),label_smoothing=0.1)

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / (num_batches)

    train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
    dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, config, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # saved = torch.load(args.filepath)
    # saved = torch.load(args.filepath, weights_only=False)
    # config = saved['model_config']
    # model = GPT2SentimentClassifier(config)
    # model.load_state_dict(saved['model'])
    # model = model.to(device)




    saved = torch.load(args.filepath, weights_only=False)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    
    # === 新增：如果在 lora 模式下训练，评估时也要先套上一样的壳子才能正常 load_state_dict ===
    if config.fine_tune_mode == 'lora':
        lora_config = LoraConfig(

            target_modules=["query", "key", "value", "attention_dense", "interm_dense", "out_dense"],
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=["classifier"]
        )
        model = get_peft_model(model, lora_config)
    # ======================================================================

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

    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
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
  # parser.add_argument("--fine-tune-mode", type=str,
  #                     help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
  #                     choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
  
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: ..., full-model: ..., lora: use PEFT LoRA',
                      choices=('last-linear-layer', 'full-model', 'lora'), default="last-linear-layer")
  
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-3)
  
  parser.add_argument("--lora_r", type=int, default=16)
  parser.add_argument("--lora_alpha", type=int, default=32)
  parser.add_argument("--lora_dropout", type=float, default=0.2)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = get_args()
  save_dir = setup_experiment(args)
  args.save_dir = save_dir
  seed_everything(args.seed)

  print('Training Sentiment Classifier on SST...')
  config = SimpleNamespace(
      filepath=os.path.join(save_dir, 'sst-all-classifier.pt'),
      lr=args.lr,
      use_gpu=args.use_gpu,
      epochs=args.epochs,
      batch_size=args.batch_size,
      hidden_dropout_prob=args.hidden_dropout_prob,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=args.fine_tune_mode,
      dev_out=os.path.join(save_dir, 'predictions', 'sst-dev.csv'),
      test_out=os.path.join(save_dir, 'predictions', 'sst-test.csv'),
        # === 新增以下4行，将全局 args 的参数传进去 ===
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_dir=args.save_dir 
  )

  train(config)

  print('Evaluating on SST...')
  test(config)

  print('Training Sentiment Classifier on cfimdb...')
  config = SimpleNamespace(
    filepath=os.path.join(save_dir, 'cfimdb-all-classifier.pt'),
    lr=args.lr,
    use_gpu=args.use_gpu,
    epochs=args.epochs,
    batch_size=8,
    hidden_dropout_prob=args.hidden_dropout_prob,
    train='data/ids-cfimdb-train.csv',
    dev='data/ids-cfimdb-dev.csv',
    test='data/ids-cfimdb-test-student.csv',
    fine_tune_mode=args.fine_tune_mode,
    dev_out=os.path.join(save_dir, 'predictions', 'cfimdb-dev.csv'),
    test_out=os.path.join(save_dir, 'predictions', 'cfimdb-test.csv'),
    # === 新增以下4行，将全局 args 的参数传进去 ===
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    save_dir=args.save_dir
  )

  train(config)

  print('Evaluating on cfimdb...')
  test(config)
