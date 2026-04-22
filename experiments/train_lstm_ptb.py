"""
PTB LSTM Language Model — standard setup following Merity et al. (2018) / Adan paper.

Key design decisions (expert):
1. Optimizer for baselines:
   - SGD with LR=30, grad_clip=0.25, ASGD trigger (NT-ASGD)
   - Adam/AdamW: lr=1e-3, eps=1e-8, NO cosine decay (step LR or constant)
     Actually for LSTM, Adam works best with CONSTANT lr + clipping
   - Adan: lr=1e-3, betas=(0.98,0.92,0.99)

2. LAKTJU_NS: Same as AdamW but with periodic NS orthogonalization
   - ns_interval=50 (frequent enough to make a difference on sequences)
   - grad_clip=0.25 (critical for LSTM stability)

3. Standard 2-layer LSTM medium model:
   - nhid=650, ninp=650, dropout=0.5
   - batch=20, bptt=35, epochs=100
   - Target: test PPL ~82 (SGD), ~65-70 (Adam-based)

4. Correct LR strategy for Adam on PTB:
   - Constant lr=5e-4, NO schedule (cosine makes it diverge)
   - Weight decay = 1.2e-6 (Merity default)
"""

import sys, os, math, time, argparse, json, datetime
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adan import Adan
from LAKTJU_NS import LAKTJU_NS
try:
    from LAKTJU_NS_v9 import LAKTJU_NS_v9
except ImportError:
    pass
try:
    from LAKTJU_NS_v10 import LAKTJU_NS_v10
except ImportError:
    pass

# ────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────

class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self._tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self._tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test  = self._tokenize(os.path.join(path, 'ptb.test.txt'))

    def _tokenize(self, path):
        with open(path) as f:
            tokens = sum(len(line.split()) + 1 for line in f)
        with open(path) as f:
            ids = torch.LongTensor(tokens)
            tok = 0
            for line in f:
                for word in line.split() + ['<eos>']:
                    ids[tok] = self.dictionary.add_word(word)
                    tok += 1
        return ids


def batchify(data, bsz, device):
    nb = data.size(0) // bsz
    data = data.narrow(0, 0, nb * bsz).view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    seq = min(bptt, len(source) - 1 - i)
    return source[i:i+seq], source[i+1:i+1+seq].reshape(-1)


# ────────────────────────────────────────────────────────────────
# Model  (medium LSTM, Zaremba / Merity style)
# ────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=False)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.weight = self.encoder.weight   # weight tying
        self.nhid = nhid
        self.nlayers = nlayers
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        out, hidden = self.rnn(emb, hidden)
        out = self.drop(out)
        dec = self.decoder(out.reshape(-1, self.nhid))
        return dec, hidden

    def init_hidden(self, bsz):
        w = next(self.parameters())
        return (w.new_zeros(self.nlayers, bsz, self.nhid),
                w.new_zeros(self.nlayers, bsz, self.nhid))


def repack(h):
    return (h[0].detach(), h[1].detach())


# ────────────────────────────────────────────────────────────────
# Train / Eval
# ────────────────────────────────────────────────────────────────

def evaluate(model, data, bptt, criterion, bsz):
    model.eval()
    total = 0.
    hidden = model.init_hidden(bsz)
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, bptt):
            x, y = get_batch(data, i, bptt)
            out, hidden = model(x, hidden)
            hidden = repack(hidden)
            total += len(x) * criterion(out, y).item()
    return total / (data.size(0) - 1)


def train_one_epoch(model, train_data, optimizer, criterion, bptt, bsz, grad_clip):
    model.train()
    total = 0.
    hidden = model.init_hidden(bsz)
    nbatches = (train_data.size(0) - 1) // bptt
    for i in range(0, train_data.size(0) - 1, bptt):
        x, y = get_batch(train_data, i, bptt)
        hidden = repack(hidden)
        optimizer.zero_grad()
        out, hidden = model(x, hidden)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / nbatches


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', default='LAKTJU_NS',
                   choices=['SGD', 'Adam', 'AdamW', 'Adan', 'LAKTJU_NS', 'LAKTJU_NS_v9', 'LAKTJU_NS_v10'])
    p.add_argument('--data',      default='../data/ptb')
    p.add_argument('--save_dir',  default='./results')
    p.add_argument('--epochs',    type=int,   default=100)
    p.add_argument('--seed',      type=int,   default=42)
    p.add_argument('--lr',        type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1.2e-6)
    p.add_argument('--batch_size',   type=int,   default=20)
    p.add_argument('--bptt',         type=int,   default=35)
    p.add_argument('--nhid',  type=int, default=650)
    p.add_argument('--ninp',  type=int, default=650)
    p.add_argument('--nlayers', type=int, default=2)
    p.add_argument('--dropout',   type=float, default=0.5)
    p.add_argument('--grad_clip', type=float, default=0.25)
    # LR decay for SGD (halve when no improvement)
    p.add_argument('--lr_decay',  type=float, default=4.0)
    p.add_argument('--patience',  type=int,   default=5)
    # LAKTJU_NS
    p.add_argument('--ns_interval', type=int,   default=50)
    p.add_argument('--ns_steps',    type=int,   default=3)
    p.add_argument('--ns_max_dim',  type=int,   default=1024)
    p.add_argument('--ns_strength', type=float, default=0.3,
                   help='NS blending strength for v9 (0=no NS, 1=full NS)')
    p.add_argument('--no_scheduler', action='store_true',
                   help='Disable ReduceLROnPlateau (use fixed LR)')
    return p.parse_args()


def build_optimizer(args, model):
    params = list(model.parameters())
    opt = args.optimizer

    if opt == 'SGD':
        lr = args.lr or 20.0
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=args.weight_decay)
        scheduler = None   # manual LR decay

    elif opt == 'Adam':
        lr = args.lr or 1e-3
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience)

    elif opt == 'AdamW':
        lr = args.lr or 1e-3
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                      eps=1e-8, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience)

    elif opt == 'Adan':
        lr = args.lr or 1e-3
        optimizer = Adan(params, lr=lr, betas=(0.98, 0.92, 0.99),
                        eps=1e-8, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience)

    elif opt == 'LAKTJU_NS':
        lr = args.lr or 1e-3
        optimizer = LAKTJU_NS(params, lr=lr, betas=(0.9, 0.999),
                              eps=1e-8, weight_decay=args.weight_decay,
                              ns_interval=args.ns_interval,
                              ns_steps=args.ns_steps,
                              min_ndim=2,
                              ns_max_dim=args.ns_max_dim)
        if getattr(args, 'no_scheduler', False):
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=args.patience)

    elif opt == 'LAKTJU_NS_v9':
        lr = args.lr or 1e-3
        optimizer = LAKTJU_NS_v9(params, lr=lr, betas=(0.9, 0.999),
                              eps=1e-8, weight_decay=args.weight_decay,
                              ns_interval=args.ns_interval,
                              ns_steps=args.ns_steps,
                              min_ndim=2,
                              ns_max_dim=args.ns_max_dim,
                              ns_strength=args.ns_strength)
        if getattr(args, 'no_scheduler', False):
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=args.patience)

    elif opt == 'LAKTJU_NS_v10':
        lr = args.lr or 1e-3
        optimizer = LAKTJU_NS_v10(params, lr=lr, betas=(0.9, 0.999),
                              eps=1e-8, weight_decay=args.weight_decay,
                              ns_interval=args.ns_interval,
                              ns_steps=args.ns_steps,
                              nhid=args.nhid)
        if getattr(args, 'no_scheduler', False):
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=args.patience)

    return optimizer, scheduler


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Optimizer: {args.optimizer}  lr={args.lr}  wd={args.weight_decay}", flush=True)
    print(f"Model: {args.nlayers}L LSTM nhid={args.nhid} dropout={args.dropout}", flush=True)

    corpus = Corpus(args.data)
    ntokens = len(corpus.dictionary)
    print(f"Vocab: {ntokens}  Train: {corpus.train.size(0):,} tokens", flush=True)

    train_data = batchify(corpus.train, args.batch_size, device)
    valid_data = batchify(corpus.valid, args.batch_size, device)
    test_data  = batchify(corpus.test,  args.batch_size, device)

    model = LSTMModel(ntokens, args.ninp, args.nhid, args.nlayers, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.1f}M", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer(args, model)

    best_val_loss = float('inf')
    best_test_ppl = float('inf')
    no_improve = 0
    cur_lr = optimizer.param_groups[0]['lr']

    # SGD manual decay tracking
    sgd_best_val = float('inf')
    sgd_no_improve = 0

    train_ppls, val_ppls = [], []

    print(f"\n{'Ep':>4} {'TrainPPL':>9} {'ValPPL':>9} {'TestPPL':>9} {'LR':>9}  Time", flush=True)
    print("-"*60, flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_data, optimizer, criterion,
                                     args.bptt, args.batch_size, args.grad_clip)
        val_loss = evaluate(model, valid_data, args.bptt, criterion, args.batch_size)

        train_ppl = math.exp(min(train_loss, 20))
        val_ppl   = math.exp(min(val_loss,   20))
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        # LR scheduling
        if args.optimizer == 'SGD':
            # Manual decay: halve LR when no improvement for patience epochs
            if val_loss < sgd_best_val:
                sgd_best_val = val_loss; sgd_no_improve = 0
            else:
                sgd_no_improve += 1
                if sgd_no_improve >= args.patience:
                    cur_lr /= args.lr_decay
                    for g in optimizer.param_groups: g['lr'] = cur_lr
                    sgd_no_improve = 0
        elif scheduler is not None:
            scheduler.step(val_loss)

        cur_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss = evaluate(model, test_data, args.bptt, criterion, args.batch_size)
            best_test_ppl = math.exp(min(test_loss, 20))
            torch.save(model.state_dict(), f"{args.save_dir}/lstm_ptb_{args.optimizer}_best.pt")

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch <= 5:
            print(f"{epoch:>4} {train_ppl:>9.2f} {val_ppl:>9.2f} {best_test_ppl:>9.2f} "
                  f"{cur_lr:>9.2e}  {elapsed:.1f}s", flush=True)

    # Final evaluation
    print(f"\nBest Valid PPL: {math.exp(best_val_loss):.2f}")
    print(f"Best Test  PPL: {best_test_ppl:.2f}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        'optimizer': args.optimizer,
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'ns_interval': getattr(args, 'ns_interval', None),
        'ns_steps': getattr(args, 'ns_steps', None),
        'seed': args.seed, 'epochs': args.epochs,
        'nhid': args.nhid, 'nlayers': args.nlayers,
        'best_val_ppl': math.exp(best_val_loss),
        'best_test_ppl': best_test_ppl,
        'train_ppls': train_ppls,
        'val_ppls': val_ppls,
    }
    fname = f"{args.save_dir}/ptb_lstm_{args.optimizer}_seed{args.seed}_{ts}.json"
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {fname}")
    return best_test_ppl


if __name__ == '__main__':
    main()
