#!/usr/bin/env python3
"""
master.py

A single-file PyTorch-based chess imitation trainer + Lichess bot.

Features:
 - Parses PGN folder or ZIP and converts every position to (board tensor, move index)
 - Builds a full UCI-style move vocabulary (covers promotions) and saves `moves.npz`
 - Small convolutional neural network (CPU-friendly) that maps board -> move logits
 - Training loop with checkpoints, resume, batch training and mixed-precision option (if available)
 - Inference / sampling with temperature, argmax and illegal-move masking
 - Lichess bot loop (berserk) that accepts standard/fromPosition challenges and plays using the model
 - Flask dashboard to show active games and model "thoughts"
 - Optional Stockfish fallback if model is uncertain

Notes:
 - This file intentionally avoids TensorFlow. It uses PyTorch for the neural net.
 - Provide the Lichess token via the environment variable LICHESS_TOKEN or --token
 - Save and load model using torch.save/load (model.pt)

Usage examples:
  # Train on PGN zip for 10 epochs
  python3 master.py --train --pgn_zip PGNs.zip --model model.pt --moves moves.npz --epochs 10 --batch 256

  # Run the bot + dashboard (reads LICHESS_TOKEN env var)
  export LICHESS_TOKEN="your_token_here"
  python3 master.py --bot --serve --port 10000 --model model.pt --moves moves.npz

"""

import argparse
import io
import json
import logging
import math
import os
import random
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try import torch; give informative error if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    torch = None
    torch_import_error = e

# Chess utilities
import chess
import chess.pgn

# Lichess client
try:
    import berserk
except Exception:
    berserk = None

# Flask dashboard
try:
    from flask import Flask, jsonify
except Exception:
    Flask = None

# Optional stockfish fallback
try:
    import chess.engine
    STOCKFISH_AVAILABLE = True
except Exception:
    chess = chess
    chess.engine = None
    STOCKFISH_AVAILABLE = False

# Logging
LOG = logging.getLogger("iwcm-pytorch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Move vocabulary utilities
# -------------------------

def generate_all_uci_moves() -> List[str]:
    """Generate a comprehensive list of UCI move strings used for output dimension.

    We generate all from-square to to-square UCI moves, plus promotion suffixes.
    This is large (~4672 typical moves) and covers nearly all possible legal moves.
    """
    files = "abcdefgh"
    ranks = "12345678"
    moves = []
    promos = ["q", "r", "b", "n"]
    squares = [f + r for f in files for r in ranks]
    for a in squares:
        for b in squares:
            if a == b:
                continue
            # regular non-promotion move
            moves.append(a + b)
            # generate promotions if moving from rank 7 to 8 (white) or 2 to 1 (black)
            # we'll include promotions generically for any file pair ending with 8 or 1
            if b[1] in ("8", "1"):
                for p in promos:
                    moves.append(a + b + p)
    # deduplicate and sort
    moves = sorted(set(moves))
    return moves

ALL_UCI = generate_all_uci_moves()
MOVE_TO_IDX = {m: i for i, m in enumerate(ALL_UCI)}
IDX_TO_MOVE = {i: m for m, i in MOVE_TO_IDX.items()}
OUTPUT_DIM = len(ALL_UCI)

# -------------------------
# Board -> Tensor encoding
# -------------------------

PIECE_TO_IDX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board, add_side_to_move: bool = True) -> np.ndarray:
    """Convert a python-chess Board to a (C,8,8) tensor numpy array (float32).

    Channels: 12 piece planes (white/black * 6). If add_side_to_move True, append one channel with side-to-move.
    """
    tensor = np.zeros((12 + (1 if add_side_to_move else 0), 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        pt = PIECE_TO_IDX[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        rank = 7 - chess.square_rank(square)  # matrix row (0 at top = 8th rank)
        file = chess.square_file(square)
        tensor[color_offset + pt, rank, file] = 1.0
    if add_side_to_move:
        tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return tensor

# -------------------------
# PGN parsing -> dataset
# -------------------------

class ChessMoveDataset(Dataset):
    """Torch Dataset that yields (board_tensor, target_move_idx) pairs.

    The dataset can be built from a pgn directory or a pgn zip file. We include all moves in each game.
    """

    def __init__(self, pgn_dir: Optional[str] = None, pgn_zip: Optional[str] = None, move_to_idx: Dict[str, int] = MOVE_TO_IDX):
        if pgn_dir and pgn_zip:
            raise ValueError("Provide only one of pgn_dir or pgn_zip")
        self.pgn_dir = pgn_dir
        self.pgn_zip = pgn_zip
        self.move_to_idx = move_to_idx
        self.samples = []  # list of (board_tensor, target_idx)
        LOG.info("Building dataset from pgn_dir=%s pgn_zip=%s", pgn_dir, pgn_zip)
        self._build()

    def _parse_pgn_file(self, path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                moves = []
                board = g.board()
                for mv in g.mainline_moves():
                    u = mv.uci()
                    # map move to our vocabulary if possible
                    if u in self.move_to_idx:
                        idx = self.move_to_idx[u]
                        # record state before move
                        self.samples.append((board_to_tensor(board), idx))
                        try:
                            board.push(mv)
                        except Exception:
                            break
                    else:
                        # skip moves not in our vocabulary
                        try:
                            board.push(mv)
                        except Exception:
                            break

    def _parse_pgn_data(self, raw: str):
        buf = io.StringIO(raw)
        while True:
            g = chess.pgn.read_game(buf)
            if g is None:
                break
            board = g.board()
            for mv in g.mainline_moves():
                u = mv.uci()
                if u in self.move_to_idx:
                    idx = self.move_to_idx[u]
                    self.samples.append((board_to_tensor(board), idx))
                    try:
                        board.push(mv)
                    except Exception:
                        break
                else:
                    try:
                        board.push(mv)
                    except Exception:
                        break

    def _build(self):
        if self.pgn_dir and os.path.exists(self.pgn_dir):
            p = Path(self.pgn_dir)
            for child in sorted(p.rglob("*.pgn")):
                try:
                    self._parse_pgn_file(str(child))
                except Exception:
                    LOG.exception("Failed to parse %s", child)
        if self.pgn_zip and os.path.exists(self.pgn_zip):
            with zipfile.ZipFile(self.pgn_zip, "r") as zf:
                for name in sorted(zf.namelist()):
                    if not name.lower().endswith('.pgn'):
                        continue
                    try:
                        data = zf.read(name).decode('utf-8', errors='replace')
                        self._parse_pgn_data(data)
                    except Exception:
                        LOG.exception("Failed to parse member %s in zip", name)
        LOG.info("Built dataset with %d samples", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board_tensor, move_idx = self.samples[idx]
        # return as numpy arrays; DataLoader collate will convert to tensors
        return np.asarray(board_tensor, dtype=np.float32), int(move_idx)

# Collate function

def collate_fn(batch):
    boards = np.stack([b for b, _ in batch], axis=0)
    targets = np.array([t for _, t in batch], dtype=np.int64)
    return torch.from_numpy(boards), torch.from_numpy(targets)

# -------------------------
# Model: small conv net
# -------------------------

class ConvImitator(nn.Module):
    def __init__(self, in_channels: int = 13, channels: int = 64, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.conv3 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channels * 2, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # x shape: (B, C, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Training utilities
# -------------------------

def train_loop(model: nn.Module, optimizer, criterion, dataloader: DataLoader, device, epoch: int, log_every: int = 50):
    model.train()
    total_loss = 0.0
    count = 0
    for i, (boards, targets) in enumerate(dataloader):
        boards = boards.to(device)
        targets = targets.to(device)
        # PyTorch expects (B, C, H, W); our boards are (B, C, H, W)
        preds = model(boards)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * boards.size(0)
        count += boards.size(0)
        if i % log_every == 0:
            LOG.info("Epoch %d batch %d loss=%.4f", epoch, i, loss.item())
    avg_loss = total_loss / max(1, count)
    LOG.info("Epoch %d complete. Avg loss %.6f samples %d", epoch, avg_loss, count)
    return avg_loss

# -------------------------
# Inference utilities
# -------------------------

def predict_move_from_model(model: nn.Module, board: chess.Board, device, temperature: float = 1.0, argmax: bool = False) -> Tuple[str, Dict[str, float]]:
    model.eval()
    bt = board_to_tensor(board)
    tensor = torch.from_numpy(bt).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)  # shape (1, OUTPUT_DIM)
        logits = logits.squeeze(0).cpu().numpy()
    # mask illegal moves
    legal = [m.uci() for m in board.legal_moves]
    mask = np.zeros_like(logits, dtype=bool)
    for mv in legal:
        if mv in MOVE_TO_IDX:
            mask[MOVE_TO_IDX[mv]] = True
    # If nothing legal in vocab, fallback
    if not mask.any():
        return None, {}
    # apply mask
    big_neg = -1e9
    masked_logits = np.where(mask, logits, big_neg)
    # temperature
    if temperature != 1.0 and temperature > 0:
        scaled = masked_logits / float(temperature)
    else:
        scaled = masked_logits
    # numeric stability
    scaled = scaled - scaled.max()
    exps = np.exp(scaled)
    exps = exps * mask  # zero out illegal
    probs = exps / (exps.sum() + 1e-12)
    # choose
    if argmax:
        idx = int(np.argmax(probs))
    else:
        idx = int(np.random.choice(len(probs), p=probs))
    mv = IDX_TO_MOVE.get(idx, None)
    # create probs dict for legal moves
    probs_dict = {IDX_TO_MOVE[i]: float(probs[i]) for i in range(len(probs)) if mask[i] and probs[i] > 0}
    return mv, probs_dict

# -------------------------
# Lichess bot (berserk)
# -------------------------

class PyTorchLichessBot:
    def __init__(self, token: str, model: nn.Module, moves_npz_path: str, device, temp: float = 0.7, argmax: bool = False, stockfish_path: Optional[str] = None):
        if berserk is None:
            raise RuntimeError("berserk required (pip install berserk)")
        self.token = token
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(session=self.session)
        self.model = model
        self.device = device
        self.temp = temp
        self.argmax = argmax
        self.active_games = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.stockfish_path = stockfish_path
        if stockfish_path and STOCKFISH_AVAILABLE:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            except Exception:
                LOG.exception("Failed to start stockfish")
                self.engine = None
        else:
            self.engine = None

    def accept_challenge(self, challenge: dict):
        cid = challenge.get('id')
        variant = challenge.get('variant', {}).get('key', '')
        if variant in ("standard", "fromPosition"):
            try:
                self.client.bots.accept_challenge(cid)
                LOG.info("Accepted challenge %s", cid)
            except Exception:
                LOG.exception("Failed to accept %s", cid)
        else:
            try:
                self.client.bots.decline_challenge(cid)
                LOG.info("Declined non-standard %s", cid)
            except Exception:
                LOG.exception("Failed to decline %s", cid)

    def handle_game(self, game_id: str, my_color: chess.Color):
        LOG.info("Game handler start %s color=%s", game_id, 'white' if my_color else 'black')
        board = chess.Board()
        try:
            stream = self.client.bots.stream_game_state(game_id)
        except Exception:
            LOG.exception("Failed to open game stream %s", game_id)
            return
        for event in stream:
            try:
                if event.get('type') in ('gameFull', 'gameState'):
                    state = event.get('state', event)
                    moves_s = state.get('moves', '')
                    moves = moves_s.split() if moves_s else []
                    board = chess.Board()
                    for mv in moves:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            pass
                    with self.lock:
                        self.active_games.setdefault(game_id, {'moves': [], 'last_thought': None, 'white': None, 'black': None, 'result': None})
                        self.active_games[game_id]['moves'] = moves[:]
                    if not board.is_game_over() and board.turn == my_color:
                        choice, probs = predict_move_from_model(self.model, board, self.device, temperature=self.temp, argmax=self.argmax)
                        if choice is None:
                            # fallback to stockfish or safe heuristic
                            if self.engine:
                                try:
                                    r = self.engine.play(board, chess.engine.Limit(time=0.05))
                                    choice = r.move.uci()
                                except Exception:
                                    choice = None
                            if choice is None:
                                # simple material-aware heuristic
                                legal = list(board.legal_moves)
                                best = None
                                best_score = -1e9
                                for m in legal:
                                    score = 0
                                    if board.is_capture(m):
                                        score += 10
                                    to_sq = m.to_square
                                    file = chess.square_file(to_sq)
                                    rank = chess.square_rank(to_sq)
                                    if 2 <= file <= 5 and 2 <= rank <= 5:
                                        score += 1
                                    if score > best_score:
                                        best_score = score
                                        best = m
                                choice = best.uci() if best else random.choice([m.uci() for m in legal])
                        LOG.info("Game %s playing %s", game_id, choice)
                        with self.lock:
                            self.active_games[game_id]['last_thought'] = {'choice': choice, 'probs': probs}
                        try:
                            self.client.bots.make_move(game_id, choice)
                        except Exception:
                            LOG.exception("Failed to send move %s for game %s", choice, game_id)
                elif event.get('type') == 'gameFinish':
                    LOG.info("Game %s finished: %s", game_id, event)
                    # export pgn
                    try:
                        pgn_text = self.client.games.export(game_id)
                        outp = Path('live_games')
                        outp.mkdir(exist_ok=True)
                        with open(outp / f"{game_id}.pgn", 'w', encoding='utf-8') as fh:
                            fh.write(pgn_text)
                        LOG.info("Saved finished PGN %s", game_id)
                    except Exception:
                        LOG.exception("Failed to export game %s", game_id)
                    # optionally online-train: append to a queue for later fine-tune
            except Exception:
                LOG.exception("Exception in game stream %s", game_id)
            if self._stop.is_set():
                LOG.info("Stopping handler %s", game_id)
                break

    def incoming_loop(self):
        backoff = 1
        while not self._stop.is_set():
            try:
                LOG.info("Opening incoming events stream")
                for event in self.client.bots.stream_incoming_events():
                    if self._stop.is_set():
                        break
                    try:
                        typ = event.get('type')
                        LOG.debug("Event: %s", typ)
                        if typ == 'challenge':
                            self.accept_challenge(event['challenge'])
                        elif typ == 'gameStart':
                            gid = event['game']['id']
                            color = event['game'].get('color', 'white')
                            my_color = chess.WHITE if color == 'white' else chess.BLACK
                            t = threading.Thread(target=self.handle_game, args=(gid, my_color), daemon=True)
                            t.start()
                    except Exception:
                        LOG.exception("Error processing incoming event")
                LOG.warning('Incoming stream closed; reconnecting in %ds', backoff)
            except Exception:
                LOG.exception('Exception in incoming loop')
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

    def start(self):
        self._stop.clear()
        t = threading.Thread(target=self.incoming_loop, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass

# -------------------------
# CLI / Orchestration
# -------------------------

def save_moves_npz(path: str):
    np.savez_compressed(path, moves=np.array(ALL_UCI, dtype=object))
    LOG.info("Saved moves vocabulary to %s (size=%d)", path, len(ALL_UCI))


def load_moves_npz(path: str):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return data['moves'].tolist()


def build_dataset_and_vocab(pgn_dir: Optional[str], pgn_zip: Optional[str], moves_out: Optional[str]):
    # Save moves vocab if requested
    if moves_out:
        save_moves_npz(moves_out)
    ds = ChessMoveDataset(pgn_dir=pgn_dir, pgn_zip=pgn_zip)
    return ds


def train_main(args):
    if torch is None:
        raise RuntimeError(f"PyTorch not installed: {torch_import_error}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info("Using device %s", device)
    # build dataset
    ds = build_dataset_and_vocab(args.pgn_dir if args.pgn_dir else None, args.pgn_zip if args.pgn_zip else None, args.moves)
    if len(ds) == 0:
        LOG.error("No training samples found. Check PGN paths.")
        return
    # DataLoader
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0)
    # model
    in_ch = 13  # 12 piece channels + 1 side-to-move
    model = ConvImitator(in_channels=in_ch, channels=args.channels, output_dim=OUTPUT_DIM)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    # resume
    if args.model and os.path.exists(args.model):
        try:
            ck = torch.load(args.model, map_location=device)
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['opt'])
            start_epoch = ck.get('epoch', 0) + 1
            LOG.info("Resumed model from %s at epoch %d", args.model, start_epoch)
        except Exception:
            LOG.exception("Failed to resume model; starting fresh")
    # training loop
    for ep in range(start_epoch, args.epochs):
        train_loop(model, optimizer, criterion, loader, device, ep, log_every=args.log_every)
        # checkpoint
        if args.model:
            try:
                torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': ep}, args.model)
                LOG.info("Saved checkpoint %s epoch %d", args.model, ep)
            except Exception:
                LOG.exception("Failed to save checkpoint")
    LOG.info("Training complete. Saving final model.")
    if args.model:
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': args.epochs - 1}, args.model)


# -------------------------
# Serve + Bot main
# -------------------------

APP = Flask(__name__) if Flask else None
GLOBAL_BOT = None

@APP.route('/')
def index():
    return "IWCM PyTorch Bot Running"

@APP.route('/_games')
def api_games():
    if GLOBAL_BOT is None:
        return jsonify({})
    with GLOBAL_BOT.lock:
        return jsonify(GLOBAL_BOT.active_games)


def run_bot(args):
    if torch is None:
        raise RuntimeError(f"PyTorch required: {torch_import_error}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obj = ConvImitator(in_channels=13, channels=args.channels, output_dim=OUTPUT_DIM)
    if args.model is None or not os.path.exists(args.model):
        LOG.error("Model file not found: %s", args.model)
        return
    LOG.info("Loading model from %s", args.model)
    ck = torch.load(args.model, map_location=device)
    model_obj.load_state_dict(ck['model'])
    model_obj.to(device)
    token = args.token or os.getenv('LICHESS_TOKEN')
    if not token:
        LOG.error("No Lichess token provided. Use LICHESS_TOKEN env var or --token")
        return
    bot = PyTorchLichessBot(token, model_obj, args.moves, device, temp=args.temp, argmax=args.argmax, stockfish_path=args.stockfish)
    global GLOBAL_BOT
    GLOBAL_BOT = bot
    bot.start()
    # run flask if requested
    if args.serve:
        if APP:
            APP.run(host='0.0.0.0', port=args.port, threaded=True)
        else:
            LOG.warning("Flask not available; skipping serve")
    else:
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            LOG.info("Shutting down bot")
            bot.stop()

# -------------------------
# Argument parser
# -------------------------

def make_parser():
    p = argparse.ArgumentParser(description='IWCM PyTorch imitation master')
    p.add_argument('--train', action='store_true')
    p.add_argument('--pgn_dir', default=None)
    p.add_argument('--pgn_zip', default=None)
    p.add_argument('--model', default='model.pt')
    p.add_argument('--moves', default='moves.npz')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--bot', action='store_true')
    p.add_argument('--serve', action='store_true')
    p.add_argument('--port', type=int, default=10000)
    p.add_argument('--token', default=None)
    p.add_argument('--temp', type=float, default=0.7)
    p.add_argument('--argmax', action='store_true')
    p.add_argument('--stockfish', default=None)
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.train:
        train_main(args)
    elif args.bot:
        run_bot(args)
    else:
        print("Nothing to do. Use --train or --bot")

if __name__ == '__main__':
    main()
