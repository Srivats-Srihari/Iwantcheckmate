#!/usr/bin/env python3
"""
master_torch_style_bot.py

PyTorch-based style-conditioned chess imitation + Lichess bot.

This file provides a complete pipeline:
 - Build vocab from PGNs (UCI moves)
 - Build training samples (context -> next move)
 - Compute a style vector from a player's PGNs
 - A small PyTorch model (embedding + GRU + style conditioning)
 - Trainer with Adam
 - Lichess bot (berserk) that scores legal moves conditioned on style
 - Optional live fine-tune on games the bot plays

Usage examples (short):
 - Build vocab: python master_torch_style_bot.py --build --pgn_dir ./pgns --vocab vocab.npz
 - Build samples: python master_torch_style_bot.py --build_samples --pgn_dir ./pgns --vocab vocab.npz --samples samples.npz
 - Compute style: python master_torch_style_bot.py --compute_style --pgn_dir ./player_pgns --vocab vocab.npz --style_out style.npz
 - Train: python master_torch_style_bot.py --train --vocab vocab.npz --samples samples.npz --model model.pt --epochs 10
 - Run bot: python master_torch_style_bot.py --bot --token YOUR_TOKEN --model model.pt --vocab vocab.npz --style_out style.npz

Note: This is intended to be more practical than the tiny NumPy toy. It still keeps things reasonably lightweight to run on modest hardware.
"""

from typing import Dict, List, Optional, Tuple
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# flask optional
try:
    from flask import Flask
except Exception:
    Flask = None

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("style_bot_torch")

# -----------------------
# Defaults & hyperparams
# -----------------------
DEFAULT_SEQ_LEN = 48
EMBED_DIM = 128
STYLE_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 1
BATCH_SIZE = 128
LR = 3e-4
EPOCHS = 10

VOCAB_FILE = "vocab.npz"
SAMPLES_FILE = "samples.npz"
MODEL_FILE = "model_style.pt"
STYLE_FILE = "style.npz"

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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
# PyTorch model
# -----------------------
class StyleModel(nn.Module):
    def __init__(self, vocab_size:int, seq_len:int=DEFAULT_SEQ_LEN, emb_dim:int=EMBED_DIM, style_dim:int=STYLE_DIM, hidden:int=HIDDEN_DIM, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.style_dim = style_dim
        self.device = device

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # small GRU to keep some order information
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, num_layers=1, batch_first=True)
        # style projection to same dim
        self.style_proj = nn.Linear(style_dim, emb_dim)
        # combine position encoding and style
        self.fc1 = nn.Linear(emb_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, idx_batch: torch.LongTensor, style_batch: torch.FloatTensor):
        # idx_batch: (B, L)
        emb = self.embed(idx_batch)  # (B, L, D)
        _, h = self.gru(emb)         # h: (1, B, D)
        pooled = h.squeeze(0)        # (B, D)
        s = self.style_proj(style_batch)  # (B, D)
        comb = torch.cat([pooled, s], dim=1)  # (B, 2D)
        h1 = F.gelu(self.fc1(comb))
        logits = self.fc2(h1)  # (B, V)
        return logits

    def save(self, path: str = MODEL_FILE):
        torch.save({
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'emb_dim': self.emb_dim,
            'style_dim': self.style_dim,
            'hidden': self.fc1.out_features,
        }, path)
        LOG.info("Saved model to %s", path)

    @staticmethod
    def load(path: str, device='cpu'):
        ckpt = torch.load(path, map_location=device)
        m = StyleModel(vocab_size=int(ckpt['vocab_size']), seq_len=int(ckpt['seq_len']), emb_dim=int(ckpt['emb_dim']) if 'emb_dim' in ckpt else EMBED_DIM, style_dim=int(ckpt.get('style_dim', STYLE_DIM)), hidden=int(ckpt.get('hidden', HIDDEN_DIM)), device=device)
        m.load_state_dict(ckpt['state_dict'])
        LOG.info("Loaded model %s", path)
        return m

# -----------------------
# Loss & training utilities
# -----------------------

def cross_entropy_loss_and_grad(logits, targets):
    loss = F.cross_entropy(logits, targets)
    return loss

# -----------------------
# Style vector computation
# -----------------------

def compute_style_from_pgns(pgn_zip: Optional[str], pgn_dir: Optional[str], vocab: Dict[str,int], style_dim:int = STYLE_DIM) -> np.ndarray:
    freq = collections.Counter()
    total = 0
    if pgn_zip:
        with zipfile.ZipFile(pgn_zip, 'r') as zf:
            for name in zf.namelist():
                if not name.lower().endswith('.pgn'): continue
                raw = zf.read(name).decode('utf-8', errors='replace')
                fh = io.StringIO(raw)
                while True:
                    g = chess.pgn.read_game(fh)
                    if g is None: break
                    for mv in g.mainline_moves():
                        freq[mv.uci()] += 1
                        total += 1
    if pgn_dir:
        p = Path(pgn_dir)
        for fn in p.rglob('*.pgn'):
            try:
                with open(fn, 'r', encoding='utf-8', errors='replace') as fh:
                    while True:
                        g = chess.pgn.read_game(fh)
                        if g is None: break
                        for mv in g.mainline_moves():
                            freq[mv.uci()] += 1
                            total += 1
            except Exception:
                LOG.exception("Error %s", fn)
    if total == 0:
        LOG.warning('No moves found to compute style; returning random vector')
        vec = np.random.randn(style_dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    vocab_size = len(vocab)
    vec_freq = np.zeros(vocab_size, dtype=np.float32)
    for m,cnt in freq.items():
        if m in vocab:
            vec_freq[vocab[m]] = cnt
        else:
            vec_freq[ vocab.get('<UNK>', 1) ] += cnt
    vec_freq = vec_freq / (np.sum(vec_freq) + 1e-12)
    rng = np.random.RandomState(12345)
    proj = rng.randn(vocab_size, style_dim).astype(np.float32) * (1.0 / math.sqrt(style_dim))
    style_vec = vec_freq.dot(proj)
    style_vec = style_vec / (np.linalg.norm(style_vec) + 1e-9)
    return style_vec.astype(np.float32)

# -----------------------
# Trainer
# -----------------------
class Trainer:
    def __init__(self, model: StyleModel, samples_path: str, style_vec: np.ndarray, batch: int = BATCH_SIZE, lr: float = LR, device='cpu'):
        self.device = device
        self.model = model.to(device)
        self.data = np.load(samples_path)
        self.X = self.data['X']
        self.y = self.data['y']
        self.batch = batch
        self.style_vec = style_vec
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        LOG.info("Trainer loaded: X=%s y=%s", self.X.shape, self.y.shape)

    def iterate_batches(self):
        n = len(self.y)
        idx = np.arange(n)
        np.random.shuffle(idx)
        for i in range(0, n, self.batch):
            bidx = idx[i:i+self.batch]
            yield self.X[bidx], self.y[bidx]

    def train(self, epochs:int=EPOCHS, save_path:Optional[str]=None, print_every:int=20):
        for ep in range(1, epochs+1):
            total_loss = 0.0
            it = 0
            t0 = time.time()
            for Xb, yb in self.iterate_batches():
                it += 1
                xb = torch.from_numpy(Xb).long().to(self.device)
                yb = torch.from_numpy(yb).long().to(self.device)
                sb = torch.from_numpy(np.tile(self.style_vec[None,:], (xb.shape[0],1))).float().to(self.device)
                self.opt.zero_grad()
                logits = self.model(xb, sb)
                loss = cross_entropy_loss_and_grad(logits, yb)
                loss.backward()
                self.opt.step()
                total_loss += float(loss.item())
                if it % print_every == 0:
                    LOG.info("Epoch %d it %d loss %.6f", ep, it, float(loss.item()))
            t1 = time.time()
            LOG.info("Epoch %d done avg_loss=%.6f time=%.1fs", ep, total_loss/max(1,it), t1-t0)
            if save_path:
                torch.save({'state_dict': self.model.state_dict(), 'vocab_size': self.model.vocab_size, 'seq_len': self.model.seq_len}, save_path)

# -----------------------
# Lichess bot
# -----------------------
class StyleLichessBot:
    def __init__(self, token: str, model: StyleModel, vocab: Dict[str,int], idx_to_move: Dict[int,str], style_vec: np.ndarray, fine_tune: bool = False, device='cpu'):
        if berserk is None:
            raise RuntimeError("berserk required: pip install berserk")
        self.token = token
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(session=self.session)
        self.model = model.to(device)
        self.vocab = vocab
        self.idx_to_move = idx_to_move
        self.move_to_idx = {m:i for m,i in vocab.items()}
        self.style_vec = torch.from_numpy(style_vec).float().to(device)
        self.fine_tune = fine_tune
        self.recent_games_buffer = []
        self.lock = threading.Lock()
        self.active_games = {}
        self.device = device

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
        legal = list(board.legal_moves)
        if len(legal) == 0:
            return None
        # build batch of candidate contexts
        seqs = []
        cand_indices = []
        for mv in legal:
            cand = mv.uci()
            seq = history_moves + [cand]
            seq_ids = [ self.move_to_idx.get(m, self.move_to_idx.get("<UNK>", 1)) for m in seq[-self.model.seq_len:] ]
            if len(seq_ids) < self.model.seq_len:
                seq_ids = [0]*(self.model.seq_len - len(seq_ids)) + seq_ids
            seqs.append(seq_ids)
            cand_indices.append(self.move_to_idx.get(cand, self.move_to_idx.get("<UNK>",1)))
        xb = torch.from_numpy(np.array(seqs, dtype=np.int64)).long().to(self.device)  # (N, L)
        sb = self.style_vec.unsqueeze(0).repeat(xb.shape[0], 1)  # (N, S)
        with torch.no_grad():
            logits = self.model(xb, sb)  # (N, V)
            probs = F.softmax(logits, dim=-1)  # (N, V)
            # score each candidate by the prob assigned to its own index
            scores = probs[torch.arange(len(cand_indices)), torch.tensor(cand_indices, device=self.device)]
            best_idx = int(torch.argmax(scores).item())
            chosen = legal[best_idx].uci()
        return chosen

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
                    if self.fine_tune:
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

    def fine_tune_worker(self, lr=1e-4):
        LOG.info("Fine-tune worker started")
        while True:
            time.sleep(60)
            with self.lock:
                if len(self.recent_games_buffer) < 64:
                    continue
                sample = random.sample(self.recent_games_buffer, min(512, len(self.recent_games_buffer)))
            X = torch.from_numpy(np.array([s[0] for s in sample], dtype=np.int64)).long().to(self.device)
            y = torch.from_numpy(np.array([s[1] for s in sample], dtype=np.int64)).long().to(self.device)
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            B = 64
            n = len(y)
            for start in range(0, n, B):
                xb = X[start:start+B]
                yb = y[start:start+B]
                sb = self.style_vec.unsqueeze(0).repeat(xb.shape[0], 1)
                logits = self.model(xb, sb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()
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
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    LOG.info("Using device: %s", device)
    if args.build:
        vocab = generate_move_vocab_from_pgns(pgn_zip=args.pgn_zip, pgn_dir=args.pgn_dir)
        save_vocab(vocab, args.vocab)
        save_idx_to_move(vocab, args.vocab.replace('.npz', '.inv.npz'))
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
        idx_to_move = load_idx_to_move(args.vocab.replace('.npz', '.inv.npz'))
        d = np.load(args.style_out, allow_pickle=True)
        style = d['style']
        model = StyleModel(vocab_size=len(vocab), seq_len=args.seq_len, emb_dim=EMBED_DIM, style_dim=STYLE_DIM, hidden=HIDDEN_DIM, device=device)
        trainer = Trainer(model, samples_path=args.samples, style_vec=style, batch=args.batch, lr=args.lr, device=device)
        trainer.train(epochs=args.epochs, save_path=args.model)
    if args.bot:
        if not os.path.exists(args.model):
            LOG.error("Model missing: %s", args.model)
            return
        vocab = load_vocab(args.vocab)
        idx_to_move = load_idx_to_move(args.vocab.replace('.npz', '.inv.npz'))
        model = StyleModel(vocab_size=len(vocab), seq_len=args.seq_len, emb_dim=EMBED_DIM, style_dim=STYLE_DIM, hidden=HIDDEN_DIM, device=device)
        model.load_state_dict(torch.load(args.model, map_location=device)['state_dict'])
        d = np.load(args.style_out, allow_pickle=True)
        style = d['style']
        token = args.token or os.getenv('LICHESS_TOKEN')
        if token is None:
            LOG.error('Provide token via --token or env LICHESS_TOKEN')
            return
        bot = StyleLichessBot(token, model, vocab, idx_to_move, style_vec=style, fine_tune=args.fine_tune, device=device)
        bot.start(serve=args.serve, port=args.port)

if __name__ == '__main__':
    main()
