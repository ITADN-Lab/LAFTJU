#!/usr/bin/env python3
"""
LSTM Language Model on Penn Treebank (PTB).

Follows the exact AdaBelief/Adan experimental setting:
- 1/2/3-layer LSTM language model
- Penn Treebank dataset
- Metric: test perplexity (lower is better)

Reference results from Adan paper (Table 10):
  1-layer: Adan 83.6, Adam 85.9, AdamW 84.7
  2-layer: Adan 65.2, Adam 67.3, AdamW 72.8
  3-layer: Adan 59.8, Adam 64.3, AdamW 69.9

Architecture (AdaBelief default for PTB):
  - Embedding size: 650 (1-layer), 650 (2-layer), 650 (3-layer)
  - Hidden size: 650 (1-layer), 650 (2-layer), 650 (3-layer)
  - Dropout: 0.5
  - Tied weights (embedding = decoder)
  - BPTT length: 35
  - Batch size: 20
  - Gradient clipping: 0.25
  - Training: 200 epochs, cosine annealing LR
"""
import os
import sys
import math
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# Add parent dir for optimizer imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from LAKTJU_NS import LAKTJU_NS
from adan import Adan


# ══════════════════════════════════════════════════════════════════════════════
# Penn Treebank Data
# ══════════════════════════════════════════════════════════════════════════════
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenize a text file."""
        assert os.path.exists(path), f'{path} not found'
        # Add words to dictionary
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
        # Tokenize
        with open(path, 'r') as f:
            ids = []
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
        return torch.tensor(ids, dtype=torch.long)


def batchify(data, bsz, device):
    """Divide data into bsz separate sequences."""
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    """Get a batch starting from position i."""
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def download_ptb(data_dir):
    """Download Penn Treebank dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, 'train.txt')
    if os.path.exists(train_path):
        return

    print("Downloading Penn Treebank dataset...")
    import urllib.request
    base_url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/'
    for split in ['train.txt', 'valid.txt', 'test.txt']:
        url = base_url + 'ptb.' + split.replace('.txt', '') + '.txt'
        out_path = os.path.join(data_dir, split)
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f'  Downloaded {split}')
        except Exception as e:
            print(f'  Failed to download {split}: {e}')
            # Try alternative URL
            alt_url = f'https://data.deepai.org/ptb.{split.replace(".txt", "")}.txt'
            try:
                urllib.request.urlretrieve(alt_url, out_path)
                print(f'  Downloaded {split} (alt)')
            except:
                raise RuntimeError(f'Cannot download PTB data. Please manually place train.txt/valid.txt/test.txt in {data_dir}')


# ══════════════════════════════════════════════════════════════════════════════
# LSTM Language Model
# ══════════════════════════════════════════════════════════════════════════════
class LSTMModel(nn.Module):
    """LSTM language model with tied weights."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=True):
        super().__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout if nlayers > 1 else 0,
                           batch_first=False)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using tied weights, nhid must equal ninp')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, train_data, criterion, optimizer, bptt, clip, device, log_interval=200):
    model.train()
    total_loss = 0.
    ntokens = model.ntoken
    hidden = model.init_hidden(train_data.size(1))
    n_batches = 0

    for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        # Detach hidden to prevent BPTT across segments
        hidden = tuple(h.detach() for h in hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, eval_data, criterion, bptt):
    model.eval()
    total_loss = 0.
    ntokens = model.ntoken
    hidden = model.init_hidden(eval_data.size(1))
    n_tokens = 0

    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            hidden = tuple(h.detach() for h in hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            total_loss += loss.item() * targets.numel()
            n_tokens += targets.numel()

    return total_loss / n_tokens


def run_experiment(args):
    """Run a single LSTM experiment."""
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    download_ptb(args.data_dir)
    corpus = Corpus(args.data_dir)
    ntokens = len(corpus.dictionary)

    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, args.eval_batch_size, device)
    test_data = batchify(corpus.test, args.eval_batch_size, device)

    # Model
    model = LSTMModel(
        ntoken=ntokens,
        ninp=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=args.dropout,
        tie_weights=args.tied
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.nlayers}-layer LSTM, {n_params:,} params, vocab={ntokens}')

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
    elif args.optimizer == 'Adan':
        optimizer = Adan(model.parameters(), lr=args.lr,
                          betas=(0.02, 0.08, 0.01),
                          weight_decay=args.weight_decay)
    elif args.optimizer == 'LAKTJU_NS':
        # Collect RNN recurrent weight params to skip NS if requested
        ns_skip_params = []
        if hasattr(args, 'ns_skip_rnn') and args.ns_skip_rnn:
            for name, param in model.named_parameters():
                if 'rnn' in name and 'weight_hh' in name:
                    ns_skip_params.append(param)
        optimizer = LAKTJU_NS(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=args.weight_decay,
                               ns_interval=args.ns_interval,
                               ns_steps=1,
                               ns_max_dim=args.ns_max_dim,
                               min_ndim=2,
                               grad_centralization=args.grad_centralization,
                               ns_skip_params=ns_skip_params)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    # LR scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)
    elif args.scheduler == 'multistep':
        # AdaBelief default: decay at epoch 100, 145
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 145], gamma=0.1)
    else:
        scheduler = None

    print(f'Optimizer: {args.optimizer}, lr={args.lr}, wd={args.weight_decay}')
    print(f'Scheduler: {args.scheduler}')
    print(f'Training: {args.epochs} epochs, bptt={args.bptt}, batch={args.batch_size}')

    # Training loop
    best_val_ppl = float('inf')
    best_test_ppl = float('inf')
    results = {
        'config': vars(args),
        'train_ppl': [],
        'val_ppl': [],
        'test_ppl': [],
        'best_val_ppl': float('inf'),
        'best_test_ppl': float('inf'),
    }

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        train_loss = train_epoch(model, train_data, criterion, optimizer,
                                  args.bptt, args.clip, device)
        val_loss = evaluate(model, val_data, criterion, args.bptt)
        test_loss = evaluate(model, test_data, criterion, args.bptt)

        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        test_ppl = math.exp(test_loss)

        results['train_ppl'].append(train_ppl)
        results['val_ppl'].append(val_ppl)
        results['test_ppl'].append(test_ppl)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_test_ppl = test_ppl
            results['best_val_ppl'] = best_val_ppl
            results['best_test_ppl'] = best_test_ppl

        # Scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_ppl)
        elif scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_t0
        if epoch % 10 == 0 or epoch <= 5 or epoch == args.epochs:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}/{args.epochs} | '
                  f'Train PPL: {train_ppl:7.2f} | Val PPL: {val_ppl:7.2f} | '
                  f'Test PPL: {test_ppl:7.2f} | '
                  f'Best Val: {best_val_ppl:.2f} / Test: {best_test_ppl:.2f} | '
                  f'LR: {cur_lr:.6f} | Time: {epoch_time:.1f}s',
                  flush=True)

    total_time = time.time() - t0
    results['total_time'] = total_time

    print(f'\n{"="*60}')
    print(f'{args.nlayers}-layer LSTM | {args.optimizer} | seed={args.seed}')
    print(f'Best Val PPL: {best_val_ppl:.2f}')
    print(f'Best Test PPL: {best_test_ppl:.2f}')
    print(f'Total time: {total_time/60:.1f} min')
    print(f'{"="*60}')

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    now_str = time.strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(args.save_dir,
                              f'lstm{args.nlayers}_{args.optimizer}_seed{args.seed}_{now_str}.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {save_path}')

    return best_val_ppl, best_test_ppl


def main():
    parser = argparse.ArgumentParser(description='LSTM Language Model on PTB')
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/ptb')
    parser.add_argument('--save_dir', type=str, default='./results_lstm')
    # Model
    parser.add_argument('--emsize', type=int, default=650)
    parser.add_argument('--nhid', type=int, default=650)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--tied', action='store_true', default=True)
    # Training
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--bptt', type=int, default=35)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam', 'AdamW', 'Adan', 'LAKTJU_NS'])
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'multistep', 'none'])
    # LAKTJU_NS specific
    parser.add_argument('--ns_interval', type=int, default=100)
    parser.add_argument('--ns_max_dim', type=int, default=1024)
    parser.add_argument('--grad_centralization', action='store_true', default=False)
    parser.add_argument('--no_grad_centralization', action='store_true')
    parser.add_argument('--ns_skip_rnn', action='store_true', default=False,
                        help='Skip NS orthogonalization on LSTM recurrent weights')

    args = parser.parse_args()

    # Set default LR per optimizer if not specified
    if args.lr is None:
        defaults = {
            'SGD': 30.0,
            'Adam': 0.001,
            'AdamW': 0.001,
            'Adan': 0.01,
            'LAKTJU_NS': 0.003,
        }
        args.lr = defaults[args.optimizer]

    if args.no_grad_centralization:
        args.grad_centralization = False

    run_experiment(args)


if __name__ == '__main__':
    main()
