#!/usr/bin/env python3
"""
AWD-LSTM Language Model on Penn Treebank.

Follows the exact setup from Merity et al. (2018) / AdaBelief / Adan papers:
- 3-layer LSTM, emsize=400, nhid=1150
- Weight Dropout (DropConnect) on weight_hh
- Variational Dropout (Locked Dropout)
- Embedding Dropout
- AR/TAR regularization
- Weight tying
- NT-ASGD support

Reference results (3-layer, from papers):
  SGD+ASGD: 63.7  |  Adam: 64.3  |  AdaBelief: 61.2  |  Adan: 59.8
"""
import os
import sys
import math
import time
import json
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from LAKTJU_NS import LAKTJU_NS
from LAKTJU_NS_Adam import LAKTJU_NS_Adam
from adan import Adan


# ══════════════════════════════════════════════════════════════════════════════
# AWD-LSTM Components (from Merity et al.)
# ══════════════════════════════════════════════════════════════════════════════

class LockedDropout(nn.Module):
    """Variational dropout: same mask across all time steps."""
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """Dropout entire words from the embedding layer."""
    if dropout and embed.training:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout)
        mask = mask.expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = F.embedding(words, masked_embed_weight, padding_idx,
                    embed.max_norm, embed.norm_type,
                    embed.scale_grad_by_freq, embed.sparse)
    return X


class WeightDrop(nn.Module):
    """DropConnect wrapper for recurrent weight matrices.

    Compatible with modern PyTorch (>= 2.0) by using torch.nn.utils.parametrize
    or manual weight replacement via _flat_weights approach.
    """
    def __init__(self, module, weights, dropout=0):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            if self.training:
                mask = raw_w.data.new_ones(raw_w.size()).bernoulli_(1 - self.dropout)
                w = nn.Parameter(mask / (1 - self.dropout) * raw_w)
            else:
                w = nn.Parameter(raw_w)
            self.module._parameters[name_w] = w
        # Rebuild _flat_weights for cuDNN
        if hasattr(self.module, '_flat_weights_names'):
            self.module._flat_weights = [
                getattr(self.module, wn) if hasattr(self.module, wn) else None
                for wn in self.module._flat_weights_names
            ]

    def forward(self, *args):
        self._setweights()
        return self.module(*args)


# ══════════════════════════════════════════════════════════════════════════════
# AWD-LSTM Model
# ══════════════════════════════════════════════════════════════════════════════

class AWDLSTMModel(nn.Module):
    """AWD-LSTM language model (Merity et al. 2018)."""

    def __init__(self, ntoken, ninp, nhid, nlayers,
                 dropout=0.4, dropouth=0.3, dropouti=0.65,
                 dropoute=0.1, wdrop=0.5, tie_weights=True):
        super().__init__()
        self.lockdrop = LockedDropout()
        self.idrop = dropouti
        self.hdrop = dropouth
        self.drop = dropout
        self.edrop = dropoute
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

        self.encoder = nn.Embedding(ntoken, ninp)

        # Stack single-layer LSTMs with WeightDrop
        self.rnns = []
        for l in range(nlayers):
            inp_size = ninp if l == 0 else nhid
            out_size = nhid if l != nlayers - 1 else (ninp if tie_weights else nhid)
            rnn = nn.LSTM(inp_size, out_size, 1, dropout=0)
            if wdrop:
                rnn = WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
            self.rnns.append(rnn)
        self.rnns = nn.ModuleList(self.rnns)

        self.decoder = nn.Linear(ninp if tie_weights else nhid, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input,
                               dropout=self.edrop if self.training else 0)
        emb = self.lockdrop(emb, self.idrop)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.hdrop)
                outputs.append(raw_output)

        output = self.lockdrop(raw_output, self.drop)
        outputs.append(output)

        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)

        if return_h:
            return decoded, new_hidden, raw_outputs, outputs
        return decoded, new_hidden

    def init_hidden(self, bsz):
        hidden = []
        for l in range(self.nlayers):
            weight = next(self.parameters())
            if isinstance(self.rnns[l], WeightDrop):
                rnn = self.rnns[l].module
            else:
                rnn = self.rnns[l]
            hsz = rnn.hidden_size
            hidden.append((weight.new_zeros(1, bsz, hsz),
                           weight.new_zeros(1, bsz, hsz)))
        return hidden


# ══════════════════════════════════════════════════════════════════════════════
# Data
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
        assert os.path.exists(path), f'{path} not found'
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
        with open(path, 'r') as f:
            ids = []
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
        return torch.tensor(ids, dtype=torch.long)


def batchify(data, bsz, device):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, train_data, criterion, optimizer, args, epoch):
    model.train()
    total_loss = 0.
    hidden = model.init_hidden(args.batch_size)
    n_batches = 0

    for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        hidden = [tuple(h.detach() for h in hs) for hs in hidden]

        optimizer.zero_grad()
        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(output, targets)

        # AR/TAR regularization
        loss = raw_loss
        if args.alpha:
            loss = loss + args.alpha * dropped_rnn_hs[-1].pow(2).mean()
        if args.beta:
            loss = loss + args.beta * (rnn_hs[-1][1:] - rnn_hs[-1][:-1]).pow(2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += raw_loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, data_source, criterion, bptt, batch_size):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(batch_size)
    n_batches = 0

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            total_loss += criterion(output, targets).item()
            hidden = [tuple(h.detach() for h in hs) for hs in hidden]
            n_batches += 1

    return total_loss / n_batches


def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data
    corpus = Corpus(args.data_dir)
    ntokens = len(corpus.dictionary)
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, args.eval_batch_size, device)
    test_data = batchify(corpus.test, args.eval_batch_size, device)
    print(f'Vocab: {ntokens}, Train: {train_data.size()}, Val: {val_data.size()}, Test: {test_data.size()}')

    # Model
    model = AWDLSTMModel(
        ntoken=ntokens,
        ninp=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=args.dropout,
        dropouth=args.dropouth,
        dropouti=args.dropouti,
        dropoute=args.dropoute,
        wdrop=args.wdrop,
        tie_weights=args.tied
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.nlayers}-layer AWD-LSTM, {n_params:,} params')

    criterion = nn.CrossEntropyLoss()

    # Collect params for NS skip
    ns_skip_params = []
    if hasattr(args, 'ns_skip_rnn') and args.ns_skip_rnn:
        for name, param in model.named_parameters():
            if 'weight_hh' in name:
                ns_skip_params.append(param)
                print(f'  NS skip: {name}')

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
                          betas=(0.98, 0.92, 0.99),
                          weight_decay=args.weight_decay)
    elif args.optimizer == 'LAKTJU_NS':
        optimizer = LAKTJU_NS(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=args.weight_decay,
                               ns_interval=args.ns_interval,
                               ns_steps=args.ns_steps,
                               ns_max_dim=args.ns_max_dim,
                               min_ndim=2,
                               grad_centralization=False,
                               ns_skip_params=ns_skip_params)
    elif args.optimizer == 'LAKTJU_NS_Adam':
        optimizer = LAKTJU_NS_Adam(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.999), eps=1e-8,
                                    weight_decay=args.weight_decay,
                                    ns_interval=args.ns_interval,
                                    ns_steps=args.ns_steps,
                                    ns_max_dim=args.ns_max_dim,
                                    min_ndim=2,
                                    grad_centralization=False,
                                    ns_skip_params=ns_skip_params)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    # LR scheduler
    if args.scheduler == 'multistep':
        # AdaBelief default milestones
        m1 = int(args.epochs * 0.5)
        m2 = int(args.epochs * 0.725)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[m1, m2], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'none':
        scheduler = None
    else:
        scheduler = None

    print(f'Optimizer: {args.optimizer}, lr={args.lr}, wd={args.weight_decay}')
    print(f'Scheduler: {args.scheduler}')
    print(f'Dropout: out={args.dropout}, h={args.dropouth}, i={args.dropouti}, '
          f'e={args.dropoute}, wdrop={args.wdrop}')
    print(f'AR alpha={args.alpha}, TAR beta={args.beta}')
    if 'LAKTJU' in args.optimizer:
        print(f'NS: interval={args.ns_interval}, max_dim={args.ns_max_dim}, '
              f'skip_rnn={args.ns_skip_rnn}, ns_steps={args.ns_steps}')

    # NT-ASGD support
    asgd_triggered = False
    asgd_trigger_epoch = -1
    stored_params = None
    if args.asgd:
        print(f'NT-ASGD enabled: nonmono={args.nonmono}')
        val_history = []

    # Training loop
    best_val_ppl = float('inf')
    best_test_ppl = float('inf')
    results = {
        'optimizer': args.optimizer,
        'nlayers': args.nlayers,
        'seed': args.seed,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'train_ppl': [],
        'val_ppl': [],
        'test_ppl': [],
        'best_val_ppl': float('inf'),
        'best_test_ppl': float('inf'),
    }
    if 'LAKTJU' in args.optimizer:
        results['ns_interval'] = args.ns_interval
        results['ns_skip_rnn'] = args.ns_skip_rnn
        results['ns_max_dim'] = args.ns_max_dim

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        train_loss = train_epoch(model, train_data, criterion, optimizer, args, epoch)
        val_loss = evaluate(model, val_data, criterion, args.bptt, args.eval_batch_size)
        test_loss = evaluate(model, test_data, criterion, args.bptt, args.eval_batch_size)

        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        test_ppl = math.exp(test_loss)

        results['train_ppl'].append(round(train_ppl, 2))
        results['val_ppl'].append(round(val_ppl, 2))
        results['test_ppl'].append(round(test_ppl, 2))

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_test_ppl = test_ppl
            results['best_val_ppl'] = round(best_val_ppl, 2)
            results['best_test_ppl'] = round(best_test_ppl, 2)

        # NT-ASGD trigger
        if args.asgd and not asgd_triggered:
            val_history.append(val_loss)
            if len(val_history) > args.nonmono and \
               val_loss > min(val_history[:-args.nonmono]):
                print(f'  => NT-ASGD triggered at epoch {epoch}')
                asgd_triggered = True
                asgd_trigger_epoch = epoch
                # Switch to ASGD
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr,
                                              t0=0, lambd=0.,
                                              weight_decay=args.weight_decay)
                scheduler = None

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_t0
        if epoch % max(1, args.epochs // 20) == 0 or epoch <= 5 or epoch == args.epochs:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:4d}/{args.epochs} | '
                  f'Train PPL: {train_ppl:7.2f} | Val PPL: {val_ppl:7.2f} | '
                  f'Test PPL: {test_ppl:7.2f} | '
                  f'Best Val: {best_val_ppl:.2f} / Test: {best_test_ppl:.2f} | '
                  f'LR: {cur_lr:.6f} | Time: {epoch_time:.1f}s',
                  flush=True)

    total_time = time.time() - t0
    results['total_time'] = total_time

    print(f'\n{"="*60}')
    print(f'{args.nlayers}-layer AWD-LSTM | {args.optimizer} | seed={args.seed}')
    print(f'Best Val PPL: {best_val_ppl:.2f}')
    print(f'Best Test PPL: {best_test_ppl:.2f}')
    print(f'Total time: {total_time/60:.1f} min')
    print(f'{"="*60}')

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Compact summary
    compact = {
        'optimizer': args.optimizer,
        'nlayers': args.nlayers,
        'seed': args.seed,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'best_val_ppl': round(best_val_ppl, 2),
        'best_test_ppl': round(best_test_ppl, 2),
    }
    if 'LAKTJU' in args.optimizer:
        compact['ns_interval'] = args.ns_interval
        compact['ns_skip_rnn'] = args.ns_skip_rnn
        compact['ns_max_dim'] = args.ns_max_dim
        compact['ns_steps'] = args.ns_steps
    compact_path = os.path.join(args.save_dir,
        f'awd{args.nlayers}_{args.optimizer}_lr{args.lr}_wd{args.weight_decay}'
        f'_ns{getattr(args, "ns_interval", 0)}_maxd{getattr(args, "ns_max_dim", 0)}'
        f'_seed{args.seed}.json')
    with open(compact_path, 'w') as f:
        json.dump(compact, f, indent=2)

    # Full results
    full_path = os.path.join(args.save_dir,
        f'awd{args.nlayers}_{args.optimizer}_seed{args.seed}_{now_str}.json')
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {full_path}')

    return best_val_ppl, best_test_ppl


def main():
    parser = argparse.ArgumentParser(description='AWD-LSTM Language Model on PTB')
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/ptb')
    parser.add_argument('--save_dir', type=str, default='./results_lstm_v3')
    # Model (AWD-LSTM defaults)
    parser.add_argument('--emsize', type=int, default=400)
    parser.add_argument('--nhid', type=int, default=1150)
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--dropouth', type=float, default=0.3)
    parser.add_argument('--dropouti', type=float, default=0.65)
    parser.add_argument('--dropoute', type=float, default=0.1)
    parser.add_argument('--wdrop', type=float, default=0.5)
    parser.add_argument('--tied', action='store_true', default=True)
    # Training
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--bptt', type=int, default=70)
    parser.add_argument('--epochs', type=int, default=750)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    # Regularization
    parser.add_argument('--alpha', type=float, default=2.0, help='AR regularization')
    parser.add_argument('--beta', type=float, default=1.0, help='TAR regularization')
    # NT-ASGD
    parser.add_argument('--asgd', action='store_true', default=False)
    parser.add_argument('--nonmono', type=int, default=5)
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam', 'AdamW', 'Adan',
                                 'LAKTJU_NS', 'LAKTJU_NS_Adam'])
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=1.2e-6)
    parser.add_argument('--scheduler', type=str, default='none',
                        choices=['cosine', 'multistep', 'none'])
    # LAKTJU_NS specific
    parser.add_argument('--ns_interval', type=int, default=100)
    parser.add_argument('--ns_max_dim', type=int, default=1024)
    parser.add_argument('--ns_skip_rnn', action='store_true', default=False)
    parser.add_argument('--ns_steps', type=int, default=1)

    args = parser.parse_args()

    # Default LR per optimizer
    if args.lr is None:
        defaults = {
            'SGD': 30.0,
            'Adam': 1e-3,
            'AdamW': 1e-3,
            'Adan': 1e-3,
            'LAKTJU_NS': 1e-3,
            'LAKTJU_NS_Adam': 1e-3,
        }
        args.lr = defaults[args.optimizer]

    run_experiment(args)


if __name__ == '__main__':
    main()
