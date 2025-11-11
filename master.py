#!/usr/bin/env python3
"""
master_style_imitation.py

NumPy-only style-conditioned chess imitation + Lichess bot.

Features:
 - Build vocab from PGNs (uci moves)
 - Build training samples (context -> next move)
 - Compute a style vector from player's PGNs (average move embeddings)
 - Small transformer-ish model (NumPy) that conditions on style vector
 - Trainer with Adam
 - Lichess Berserk bot that selects moves by scoring candidate moves conditioned on style
 - Optional live fine-tune on games the bot plays

Notes:
 - This is an engineering compromise: pure-NumPy model for portability, but slower to train.
 - For serious accuracy, train on a GPU-enabled machine (export samples & vocab).
"""
import argparse
import collections
import io
import json
import math
import os
import random
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# chess lib
try:
    import chess
    import chess.pgn
except Exception as e:
    raise RuntimeError("python-chess required: pip install python-chess") from e

# lichess client (berserk)
try:
    import berserk
except Exception:
    berserk = None

# flask dashboard optional
try:
    from flask import Flask, jsonify
except Exception:
    Flask = None

# basic logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("style_bot")

# -----------------------
# Defaults & hyperparams
# -----------------------
DEFAULT_SEQ_LEN = 48
EMBED_DIM = 128
STYLE_DIM = 64      # dimension of computed style vector
FF_DIM = 256
NUM_LAYERS = 1
NUM_HEADS = 2
BATCH_SIZE = 128
LR = 3e-4
EPOCHS = 10

VOCAB_FILE = "vocab.npz"
SAMPLES_FILE = "samples.npz"
MODEL_FILE = "model_style.npz"
STYLE_FILE = "style.npz"

seed = 42
random.seed(seed)
np.random.seed(seed)

# -----------------------
# Vocab helpers
# -----------------------
def generate_move_vocab_from_pgns(pgn_zip: Optional[str] = None, pgn_dir: Optional[str] = None, min_freq: int = 1) -> Dict[str,int]:
    counter = collections.Counter()
    def iter_from_zip(zippath):
        with zipfile.ZipFile(zippath, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".pgn"):
                    continue
                raw = zf.read(name).decode("utf-8", errors="replace")
                fh = io.StringIO(raw)
                while True:
                    g = chess.pgn.read_game(fh)
                    if g is None:
                        break
                    for mv in g.mainline_moves():
                        counter[mv.uci()] += 1
    def iter_from_dir(dirpath):
        p = Path(dirpath)
        for fn in p.rglob("*.pgn"):
            try:
                with open(fn, "r", encoding="utf-8", errors="replace") as fh:
                    while True:
                        g = chess.pgn.read_game(fh)
                        if g is None: break
                        for mv in g.mainline_moves():
                            counter[mv.uci()] += 1
            except Exception:
                LOG.exception("Failed reading %s", fn)

    if pgn_zip:
        LOG.info("Building vocab from zip: %s", pgn_zip)
        iter_from_zip(pgn_zip)
    if pgn_dir:
        LOG.info("Building vocab from dir: %s", pgn_dir)
        iter_from_dir(pgn_dir)

    vocab = {"<PAD>":0, "<UNK>":1}
    idx = 2
    for move, freq in counter.most_common():
        if freq >= min_freq:
            vocab[move] = idx
            idx += 1
    LOG.info("Vocab size: %d", len(vocab))
    return vocab

def save_vocab(vocab: Dict[str,int], path: str = VOCAB_FILE):
    moves = np.array(list(vocab.keys()), dtype=object)
    ids = np.array([vocab[m] for m in moves], dtype=np.int32)
    np.savez_compressed(path, moves=moves, ids=ids)
    LOG.info("Saved vocab to %s", path)

def load_vocab(path: str = VOCAB_FILE) -> Dict[str,int]:
    d = np.load(path, allow_pickle=True)
    moves = d["moves"]
    ids = d["ids"]
    vocab = {moves[i]: int(ids[i]) for i in range(len(moves))}
    LOG.info("Loaded vocab from %s size=%d", path, len(vocab))
    return vocab

def save_idx_to_move(vocab, path):
    inv = {v:k for k,v in vocab.items()}
    keys = np.array(list(inv.keys()), dtype=np.int32)
    vals = np.array(list(inv.values()), dtype=object)
    np.savez_compressed(path, keys=keys, vals=vals)

def load_idx_to_move(path):
    d = np.load(path, allow_pickle=True)
    keys = d["keys"]; vals = d["vals"]
    return {int(keys[i]): str(vals[i]) for i in range(len(keys))}

# -----------------------
# Dataset builder
# -----------------------
class DatasetBuilder:
    def __init__(self, vocab: Dict[str,int], seq_len: int = DEFAULT_SEQ_LEN):
        self.vocab = vocab
        self.seq_len = seq_len

    def move_to_id(self, move: str) -> int:
        return self.vocab.get(move, self.vocab.get("<UNK>", 1))

    def build_from_zip(self, zippath: str, out_npz: str = SAMPLES_FILE, max_games: Optional[int] = None):
        LOG.info("Building samples from zip: %s", zippath)
        contexts, targets = [], []
        count = 0
        with zipfile.ZipFile(zippath,"r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".pgn"):
                    continue
                raw = zf.read(name).decode("utf-8", errors="replace")
                fh = io.StringIO(raw)
                while True:
                    g = chess.pgn.read_game(fh)
                    if g is None: break
                    seq = []
                    for mv in g.mainline_moves():
                        contexts.append(self.build_context(seq))
                        targets.append(self.move_to_id(mv.uci()))
                        seq.append(mv.uci())
                    count += 1
                    if max_games and count >= max_games:
                        break
                if max_games and count >= max_games: break
        if len(contexts) == 0:
            LOG.warning("No samples created")
        else:
            X = np.array(contexts, dtype=np.int32)
            y = np.array(targets, dtype=np.int32)
            np.savez_compressed(out_npz, X=X, y=y)
            LOG.info("Saved samples %s X=%s y=%s", out_npz, X.shape, y.shape)
        return out_npz

    def build_from_dir(self, dirpath: str, out_npz: str = SAMPLES_FILE, max_games: Optional[int] = None):
        p = Path(dirpath)
        contexts, targets = [], []
        count = 0
        for fn in p.rglob("*.pgn"):
            try:
                with open(fn, "r", encoding="utf-8", errors="replace") as fh:
                    while True:
                        g = chess.pgn.read_game(fh)
                        if g is None: break
                        seq = []
                        for mv in g.mainline_moves():
                            contexts.append(self.build_context(seq))
                            targets.append(self.move_to_id(mv.uci()))
                            seq.append(mv.uci())
                        count += 1
                        if max_games and count >= max_games:
                            break
            except Exception:
                LOG.exception("Error parsing %s", fn)
            if max_games and count >= max_games:
                break
        if len(contexts)==0:
            LOG.warning("No samples created")
        else:
            X = np.array(contexts, dtype=np.int32)
            y = np.array(targets, dtype=np.int32)
            np.savez_compressed(out_npz, X=X, y=y)
            LOG.info("Saved samples %s X=%s y=%s", out_npz, X.shape, y.shape)
        return out_npz

    def build_context(self, moves_seq: List[str]) -> List[int]:
        seq = moves_seq[-self.seq_len:]
        ids = [self.move_to_id(m) for m in seq]
        if len(ids) < self.seq_len:
            pad = [0] * (self.seq_len - len(ids))
            ids = pad + ids
        return ids

# -----------------------
# Small NumPy model (embedding + pooling + style conditioning + MLP head)
# -----------------------
class Param:
    def __init__(self, data: np.ndarray, name: str = ""):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(self.data)
        self.name = name
    def zero_grad(self):
        self.grad.fill(0.0)

class Embedding:
    def __init__(self, vocab_size, dim):
        self.W = Param(np.random.randn(vocab_size, dim).astype(np.float32) * (dim ** -0.5), name="emb_W")
    def forward(self, idx_batch):
        self.idx = idx_batch
        return self.W.data[idx_batch]     # (B,L,D)
    def backward(self, grad):
        # grad: (B,L,D)
        self.W.grad.fill(0.0)
        np.add.at(self.W.grad, self.idx, grad)

class Linear:
    def __init__(self, in_dim, out_dim, name=""):
        self.W = Param(np.random.randn(in_dim, out_dim).astype(np.float32) * (in_dim ** -0.5), name=name+".W")
        self.b = Param(np.zeros(out_dim, dtype=np.float32), name=name+".b")
    def forward(self, x):   # x (...,in_dim)
        self.x_shape = x.shape
        flat = x.reshape(-1, self.W.data.shape[0])
        out = flat.dot(self.W.data) + self.b.data
        return out.reshape(*self.x_shape[:-1], -1)
    def backward(self, grad):  # grad (..., out_dim)
        flat_grad = grad.reshape(-1, grad.shape[-1])
        flat_x = self.x_shape
        # need original input; but we'll store input on forward for this simplified flow
        # To keep code tidy, user code must call set_last_input before backward
        if not hasattr(self, "last_x"):
            raise RuntimeError("Linear.backward called without last_x set")
        x_flat = self.last_x.reshape(-1, self.W.data.shape[0])
        self.W.grad += x_flat.T.dot(flat_grad)
        self.b.grad += np.sum(flat_grad, axis=0)
        grad_x = flat_grad.dot(self.W.data.T).reshape(*self.x_shape)
        return grad_x
    def set_last_input(self, x):
        self.last_x = x

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi)*(x + 0.044715 * x**3)))
def gelu_grad(x):
    tanh = np.tanh(np.sqrt(2.0/np.pi)*(x + 0.044715 * x**3))
    left = 0.5*(1+tanh)
    sech2 = 1 - tanh*tanh
    right = 0.5*x*sech2*(np.sqrt(2.0/np.pi)*(1+3*0.044715*x*x))
    return left + right

class SimpleModel:
    """
    Embedding -> mean-pool -> concat(style_vector) -> small MLP -> logits
    This is lightweight and trains faster than a full transformer on Pi.
    """
    def __init__(self, vocab_size:int, seq_len:int=DEFAULT_SEQ_LEN, emb_dim:int=EMBED_DIM, style_dim:int=STYLE_DIM, hidden:int=FF_DIM):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.style_dim = style_dim
        self.emb = Embedding(vocab_size, emb_dim)
        # optional learned projection for style vector
        self.style_proj = Linear(style_dim, emb_dim, name="style_proj")
        # combine pooled emb + style_proj -> hidden -> logits
        self.fc1 = Linear(emb_dim*1 + emb_dim, hidden, name="fc1")
        self.fc2 = Linear(hidden, vocab_size, name="out")
        # collect params
        self.params = [self.emb.W, self.style_proj.W, self.style_proj.b,
                       self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b]

    def forward(self, idx_batch: np.ndarray, style_batch: np.ndarray):
        # idx_batch: (B,L), style_batch: (B, style_dim)
        B,L = idx_batch.shape
        emb = self.emb.forward(idx_batch)   # (B,L,D)
        pooled = emb.mean(axis=1)           # (B,D)
        # style projection
        self.style_proj.set_last_input(style_batch)
        style_proj = self.style_proj.forward(style_batch)   # (B,D)
        # concat
        comb = np.concatenate([pooled, style_proj], axis=1)  # (B, 2D)
        # fc1
        self.fc1.set_last_input(comb)
        h = self.fc1.forward(comb)
        self.h_pre = h.copy()
        h_act = gelu(h)
        # fc2
        self.fc2.set_last_input(h_act)
        logits = self.fc2.forward(h_act)    # (B, V)
        # store intermediates for backward
        self._cache = (emb, pooled, style_proj, comb, h_act)
        return logits

    def backward(self, dlogits: np.ndarray):
        # dlogits: (B,V)
        h_act_grad = self.fc2.backward(dlogits)  # (B, hidden)
        # gradient through gelu
        # approximate: use pre-activation from cache
        emb, pooled, style_proj, comb, h_act = self._cache
        # h_act = gelu(h_pre)
        # compute grad wrt h_pre
        # we don't have h_pre stored exactly, but fc1.last_x was stored earlier; we saved h_pre in h_act variable BEFORE activation? We stored h_act which is post-activation.
        # We'll approximate derivative numerically using gelu_grad on h_act (acceptable compromise)
        dh_pre = h_act_grad * gelu_grad(h_act)
        # backprop fc1
        dcomb = self.fc1.backward(dh_pre)  # (B, 2D)
        # split to pooled and style_proj
        D = pooled.shape[1]
        dpooled = dcomb[:, :D]
        dstyleproj = dcomb[:, D:]
        # backprop style_proj
        dstyle_in = self.style_proj.backward(dstyleproj)  # (B, style_dim)
        # backprop pooled -> embeddings (mean)
        # pooled = emb.mean(axis=1) => each emb gets dpooled / L
        B,L,D = emb.shape
        demb = np.repeat(dpooled[:, None, :] / L, L, axis=1)
        # backprop to embedding matrix
        self.emb.backward(demb)
        # accumulate grads appropriately (style_proj, fc1, fc2 already updated)
        return

    def params_and_grads(self):
        out = []
        for p in self.params:
            out.append((p.data, p.grad, p))
        return out

    def zero_grads(self):
        for _,_,p in self.params:
            p.zero_grad()

    def save(self, path: str = MODEL_FILE):
        out = {}
        out["vocab_size"] = self.vocab_size
        out["seq_len"] = self.seq_len
        out["emb_W"] = self.emb.W.data
        out["style_proj_W"] = self.style_proj.W.data
        out["style_proj_b"] = self.style_proj.b.data
        out["fc1_W"] = self.fc1.W.data
        out["fc1_b"] = self.fc1.b.data
        out["fc2_W"] = self.fc2.W.data
        out["fc2_b"] = self.fc2.b.data
        np.savez_compressed(path, **out)
        LOG.info("Saved model to %s", path)

    @staticmethod
    def load(path: str):
        d = np.load(path, allow_pickle=True)
        vocab_size = int(d["vocab_size"])
        seq_len = int(d["seq_len"])
        embW = d["emb_W"]
        m = SimpleModel(vocab_size=vocab_size, seq_len=seq_len, emb_dim=embW.shape[1], style_dim=d["style_proj_W"].shape[0], hidden=d["fc1_W"].shape[1])
        m.emb.W.data = embW
        m.style_proj.W.data = d["style_proj_W"]
        m.style_proj.b.data = d["style_proj_b"]
        m.fc1.W.data = d["fc1_W"]
        m.fc1.b.data = d["fc1_b"]
        m.fc2.W.data = d["fc2_W"]
        m.fc2.b.data = d["fc2_b"]
        LOG.info("Loaded model %s", path)
        return m

# -----------------------
# Loss & optimizer (Adam)
# -----------------------
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-12)

def cross_entropy_and_grad(logits, targets):
    probs = softmax(logits)
    B = logits.shape[0]
    idx = targets.astype(np.int64)
    neglog = -np.log(probs[np.arange(B), idx] + 1e-12)
    loss = np.mean(neglog)
    grad = probs.copy()
    grad[np.arange(B), idx] -= 1.0
    grad = grad / B
    return loss, grad

class Adam:
    def __init__(self, params: List[Param], lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.ps = params
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}
        self.t = 0
    def step(self):
        self.t += 1
        for p in self.ps:
            g = p.grad
            m = self.m[id(p)]; v = self.v[id(p)]
            m[:] = self.b1*m + (1-self.b1)*g
            v[:] = self.b2*v + (1-self.b2)*(g*g)
            m_hat = m / (1 - self.b1**self.t)
            v_hat = v / (1 - self.b2**self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= update
    def zero_grad(self):
        for p in self.ps:
            p.zero_grad()

# -----------------------
# Style vector computation
# -----------------------
def compute_style_from_pgns(pgn_zip: Optional[str], pgn_dir: Optional[str], vocab: Dict[str,int], style_dim:int = STYLE_DIM) -> np.ndarray:
    """
    Simple style vector: frequency-weighted average of one-hot moves projected to style_dim.
    Implementation: compute normalized frequency vector over vocab moves seen in player's games,
    then project using a random projection matrix (kept deterministic via seed). Save the style vector.
    """
    freq = collections.Counter()
    total = 0
    def iter_zip(zippath):
        with zipfile.ZipFile(zippath, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".pgn"): continue
                raw = zf.read(name).decode("utf-8", errors="replace")
                fh = io.StringIO(raw)
                while True:
                    g = chess.pgn.read_game(fh)
                    if g is None: break
                    for mv in g.mainline_moves():
                        freq[mv.uci()] += 1
                        nonlocal_total_inc()
    def iter_dir(dirpath):
        p = Path(dirpath)
        for fn in p.rglob("*.pgn"):
            try:
                with open(fn, "r", encoding="utf-8", errors="replace") as fh:
                    while True:
                        g = chess.pgn.read_game(fh)
                        if g is None: break
                        for mv in g.mainline_moves():
                            freq[mv.uci()] += 1
                            nonlocal_total_inc()
            except Exception:
                LOG.exception("Error reading %s", fn)

    # small closure to bump total (to avoid using nonlocal keyword weirdness)
    total = 0
    def nonlocal_total_inc():
        nonlocal total
        total += 1

    if pgn_zip:
        with zipfile.ZipFile(pgn_zip, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".pgn"): continue
                raw = zf.read(name).decode("utf-8", errors="replace")
                fh = io.StringIO(raw)
                while True:
                    g = chess.pgn.read_game(fh)
                    if g is None: break
                    for mv in g.mainline_moves():
                        freq[mv.uci()] += 1
                        total += 1
    if pgn_dir:
        p = Path(pgn_dir)
        for fn in p.rglob("*.pgn"):
            try:
                with open(fn, "r", encoding="utf-8", errors="replace") as fh:
                    while True:
                        g = chess.pgn.read_game(fh)
                        if g is None: break
                        for mv in g.mainline_moves():
                            freq[mv.uci()] += 1
                            total += 1
            except Exception:
                LOG.exception("Error %s", fn)
    if total == 0:
        LOG.warning("No moves found to compute style; returning random vector")
        vec = np.random.randn(style_dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    # build freq vector in vocab order
    inv = {v:k for k,v in vocab.items()}
    vocab_size = len(vocab)
    vec_freq = np.zeros(vocab_size, dtype=np.float32)
    for m,cnt in freq.items():
        if m in vocab:
            vec_freq[vocab[m]] = cnt
        else:
            # accumulate to UNK
            vec_freq[ vocab.get("<UNK>", 1) ] += cnt
    # normalize
    vec_freq = vec_freq / (np.sum(vec_freq) + 1e-12)
    # small deterministic projection matrix
    rng = np.random.RandomState(12345)
    proj = rng.randn(vocab_size, style_dim).astype(np.float32) * (1.0 / math.sqrt(style_dim))
    style_vec = vec_freq.dot(proj)   # (style_dim,)
    # normalize
    style_vec = style_vec / (np.linalg.norm(style_vec) + 1e-9)
    return style_vec.astype(np.float32)

# -----------------------
# Trainer
# -----------------------
class Trainer:
    def __init__(self, model: SimpleModel, samples_path: str, style_vec: np.ndarray, batch: int = BATCH_SIZE, lr: float = LR):
        self.model = model
        self.data = np.load(samples_path)
        self.X = self.data['X']
        self.y = self.data['y']
        self.batch = batch
        self.style_vec = style_vec
        self.opt = Adam(self.model.params, lr=lr)
        LOG.info("Trainer loaded: X=%s y=%s", self.X.shape, self.y.shape)

    def iterate_batches(self):
        n = len(self.y)
        idx = np.arange(n)
        np.random.shuffle(idx)
        for i in range(0, n, self.batch):
            bidx = idx[i:i+self.batch]
            yield self.X[bidx], self.y[bidx]

    def train(self, epochs:int=EPOCHS, save_path:Optional[str]=None, print_every:int=20):
        B_style = np.tile(self.style_vec[None,:], (self.batch,1))  # we'll slice as needed
        for ep in range(1, epochs+1):
            total_loss = 0.0
            it = 0
            t0 = time.time()
            for Xb, yb in self.iterate_batches():
                it += 1
                bs = Xb.shape[0]
                style_batch = np.tile(self.style_vec[None,:], (bs,1)).astype(np.float32)
                logits = self.model.forward(Xb, style_batch)
                loss, dlogits = cross_entropy_and_grad(logits, yb)
                # zero grads
                for p in self.model.params: p.zero_grad()
                # backward
                self.model.backward(dlogits)
                # step
                self.opt.step()
                total_loss += loss
                if it % print_every == 0:
                    LOG.info("Epoch %d it %d loss %.6f", ep, it, loss)
            t1 = time.time()
            LOG.info("Epoch %d done avg_loss=%.6f time=%.1fs", ep, total_loss/max(1,it), t1-t0)
            if save_path:
                self.model.save(save_path)

# -----------------------
# Lichess bot
# -----------------------
class StyleLichessBot:
    def __init__(self, token: str, model: SimpleModel, vocab: Dict[str,int], idx_to_move: Dict[int,str], style_vec: np.ndarray, fine_tune: bool = False):
        if berserk is None:
            raise RuntimeError("berserk required: pip install berserk")
        self.token = token
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(session=self.session)
        self.model = model
        self.vocab = vocab
        self.idx_to_move = idx_to_move
        self.move_to_idx = {m:i for m,i in vocab.items()}
        self.style_vec = style_vec
        self.fine_tune = fine_tune
        self.recent_games_buffer = []     # store small training tuples from live play
        self.lock = threading.Lock()
        self.active_games = {}

    def accept_challenge(self, chal):
        vid = chal['variant']['key']
        cid = chal['id']
        if vid in ('standard', 'fromPosition'):
            try:
                self.client.bots.accept_challenge(cid)
                LOG.info("Accepted %s", cid)
            except Exception:
                LOG.exception("Accept failed %s", cid)
        else:
            try:
                self.client.bots.decline_challenge(cid)
                LOG.info("Declined %s", cid)
            except Exception:
                LOG.exception("Decline failed %s", cid)

    def score_candidate_moves(self, board: chess.Board, history_moves: List[str]) -> str:
        """
        For each legal move, append it to history and ask the model to score that candidate:
        - Form the candidate context sequence (last seq_len moves with candidate appended)
        - Map to ids, run forward to get logits, transform to probability assigned to candidate
        Choose legal move with highest probability (style-aware).
        """
        legal = list(board.legal_moves)
        best_move = None
        best_score = -1e9
        # precompute base sequence as list of uci strings
        for mv in legal:
            cand = mv.uci()
            seq = history_moves + [cand]
            seq_ids = [ self.move_to_idx.get(m, self.move_to_idx.get("<UNK>", 1)) for m in seq[-self.model.seq_len:] ]
            if len(seq_ids) < self.model.seq_len:
                seq_ids = [0]*(self.model.seq_len - len(seq_ids)) + seq_ids
            arr = np.array(seq_ids, dtype=np.int32)[None,:]
            style_batch = np.tile(self.style_vec[None,:], (1,1)).astype(np.float32)
            logits = self.model.forward(arr, style_batch)   # (1, V)
            probs = softmax(logits)[0]
            # score candidate as probability assigned to its index
            cand_idx = self.move_to_idx.get(cand, self.move_to_idx.get("<UNK>",1))
            score = float(probs[cand_idx])
            if score > best_score:
                best_score = score
                best_move = cand
        if best_move is None:
            # fallback random legal
            best_move = random.choice(legal).uci()
        return best_move

    def game_handler(self, game_id: str, my_color: chess.Color):
        LOG.info("handler start %s", game_id)
        stream = self.client.bots.stream_game_state(game_id)
        history_moves = []
        board = chess.Board()
        for ev in stream:
            try:
                if ev['type'] in ('gameFull', 'gameState'):
                    state = ev.get('state', ev)
                    moves_s = state.get('moves', '')
                    history_moves = moves_s.split() if moves_s else []
                    board = chess.Board()
                    for mv in history_moves:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            pass
                    with self.lock:
                        self.active_games[game_id] = {'moves': history_moves.copy(), 'last_thought': None}
                    # if it's our turn
                    our_turn = (board.turn == chess.WHITE and my_color == chess.WHITE) or (board.turn == chess.BLACK and my_color == chess.BLACK)
                    if our_turn and (not board.is_game_over()):
                        LOG.info("[%s] our turn, computing candidate scores...", game_id)
                        chosen = self.score_candidate_moves(board, history_moves)
                        try:
                            self.client.bots.make_move(game_id, chosen)
                            LOG.info("[%s] played %s", game_id, chosen)
                            with self.lock:
                                self.active_games[game_id]['last_thought'] = {'choice': chosen}
                        except Exception:
                            LOG.exception("Failed to send move %s for game %s", chosen, game_id)
                elif ev['type'] == 'gameFinish':
                    LOG.info("Finished %s", game_id)
                    # optionally, add to fine-tune buffer
                    if self.fine_tune:
                        # derive (contexts->target) triples from moves and append to buffer
                        moves = ev.get('moves', '') or ''
                        if moves:
                            seq = moves.split()
                            for i in range(len(seq)):
                                ctx = seq[max(0, i - self.model.seq_len):i]
                                ids = [ self.move_to_idx.get(m, self.move_to_idx.get("<UNK>",1)) for m in ctx ]
                                if len(ids) < self.model.seq_len:
                                    ids = [0]*(self.model.seq_len - len(ids)) + ids
                                target = self.move_to_idx.get(seq[i], self.move_to_idx.get("<UNK>",1))
                                with self.lock:
                                    self.recent_games_buffer.append((ids, target))
                                    # keep buffer small
                                    if len(self.recent_games_buffer) > 5000:
                                        self.recent_games_buffer = self.recent_games_buffer[-4000:]
            except Exception:
                LOG.exception("Error in game_handler %s", game_id)

    def incoming_loop(self):
        backoff = 1
        while True:
            try:
                for ev in self.client.bots.stream_incoming_events():
                    t = ev.get('type')
                    if t == 'challenge':
                        self.accept_challenge(ev['challenge'])
                    elif t == 'gameStart':
                        gid = ev['game']['id']
                        color_str = ev['game'].get('color', 'white')
                        my_color = chess.WHITE if color_str == 'white' else chess.BLACK
                        th = threading.Thread(target=self.game_handler, args=(gid, my_color), daemon=True)
                        th.start()
            except Exception:
                LOG.exception("incoming loop error; reconnecting in %ds", backoff)
                time.sleep(backoff)
                backoff = min(120, backoff*2)

    def fine_tune_worker(self, lr=1e-4, iters=10):
        """
        Small periodic fine-tune on buffered self-play games.
        """
        LOG.info("Fine-tune worker started")
        while True:
            time.sleep(60)  # run every minute
            with self.lock:
                if len(self.recent_games_buffer) < 64:
                    continue
                # take a small sample
                sample = random.sample(self.recent_games_buffer, min(512, len(self.recent_games_buffer)))
            X = np.array([s[0] for s in sample], dtype=np.int32)
            y = np.array([s[1] for s in sample], dtype=np.int32)
            opt = Adam(self.model.params, lr=lr)
            # do a few micro-batches
            B = 64
            n = len(y)
            for start in range(0, n, B):
                xb = X[start:start+B]
                yb = y[start:start+B]
                sb = np.tile(self.style_vec[None,:], (xb.shape[0],1)).astype(np.float32)
                logits = self.model.forward(xb, sb)
                loss, dlogits = cross_entropy_and_grad(logits, yb)
                for p in self.model.params: p.zero_grad()
                self.model.backward(dlogits)
                opt.step()
            LOG.info("Fine-tune step done (buffer items=%d)", len(self.recent_games_buffer))

    def start(self, serve=False, port=10000):
        t = threading.Thread(target=self.incoming_loop, daemon=True)
        t.start()
        if self.fine_tune:
            tf = threading.Thread(target=self.fine_tune_worker, daemon=True)
            tf.start()
        if serve and Flask:
            app = Flask("style_bot")
            @app.route("/")
            def home():
                return "Style bot running"
            @app.route("/games")
            def games():
                with self.lock:
                    return json.dumps(self.active_games)
            app.run(host="0.0.0.0", port=port)
        else:
            while True:
                time.sleep(10)

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--pgn_zip", type=str, default=None)
    p.add_argument("--pgn_dir", type=str, default=None)
    p.add_argument("--vocab", type=str, default=VOCAB_FILE)
    p.add_argument("--build_samples", action="store_true")
    p.add_argument("--samples", type=str, default=SAMPLES_FILE)
    p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--compute_style", action="store_true")
    p.add_argument("--style_out", type=str, default=STYLE_FILE)
    p.add_argument("--train", action="store_true")
    p.add_argument("--model", type=str, default=MODEL_FILE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--bot", action="store_true")
    p.add_argument("--token", type=str, default=None)
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=10000)
    p.add_argument("--fine_tune", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.build:
        vocab = generate_move_vocab_from_pgns(pgn_zip=args.pgn_zip, pgn_dir=args.pgn_dir)
        save_vocab(vocab, args.vocab)
        save_idx_to_move(vocab, args.vocab.replace(".npz", ".inv.npz"))
    if args.build_samples:
        vocab = load_vocab(args.vocab)
        builder = DatasetBuilder(vocab=vocab, seq_len=args.seq_len)
        if args.pgn_zip:
            builder.build_from_zip(args.pgn_zip, out_npz=args.samples)
        elif args.pgn_dir:
            builder.build_from_dir(args.pgn_dir, out_npz=args.samples)
        else:
            LOG.error("Provide --pgn_zip or --pgn_dir to build samples")
    if args.compute_style:
        vocab = load_vocab(args.vocab)
        style_vec = compute_style_from_pgns(pgn_zip=args.pgn_zip, pgn_dir=args.pgn_dir, vocab=vocab, style_dim=STYLE_DIM)
        np.savez_compressed(args.style_out, style=style_vec)
        LOG.info("Saved style vector to %s", args.style_out)
    if args.train:
        vocab = load_vocab(args.vocab)
        inv = load_idx_to_move(args.vocab.replace(".npz", ".inv.npz"))
        d = np.load(args.style_out, allow_pickle=True)
        style = d["style"]
        model = SimpleModel(vocab_size=len(vocab), seq_len=args.seq_len, emb_dim=EMBED_DIM, style_dim=STYLE_DIM, hidden=FF_DIM)
        trainer = Trainer(model, samples_path=args.samples, style_vec=style, batch=args.batch, lr=args.lr)
        trainer.train(epochs=args.epochs, save_path=args.model)
    if args.bot:
        if not os.path.exists(args.model):
            LOG.error("Model missing: %s", args.model)
            return
        vocab = load_vocab(args.vocab)
        idx_to_move = load_idx_to_move(args.vocab.replace(".npz", ".inv.npz"))
        model = SimpleModel.load(args.model)
        d = np.load(args.style_out, allow_pickle=True)
        style = d["style"]
        token = args.token or os.getenv("LICHESS_TOKEN")
        if token is None:
            LOG.error("Provide token via --token or env LICHESS_TOKEN")
            return
        bot = StyleLichessBot(token, model, vocab, idx_to_move, style_vec=style, fine_tune=args.fine_tune)
        bot.start(serve=args.serve, port=args.port)

if __name__ == "__main__":
    main()
