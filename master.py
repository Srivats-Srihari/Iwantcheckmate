#!/usr/bin/env python3
"""
master.py

No-TensorFlow imitation Lichess bot + trainer (single-file).

Updates in this copy:
 - Adds --epochs and --checkpoint for repeated training passes (useful for long runs on Pi)
 - Checkpoint saves every N epochs
 - Same hybrid n-gram + FEN-count model (no TF/PyTorch)
 - Loads token from env or --token (do NOT commit token to git)

Usage examples:
  # Train for 2048 epochs (will checkpoint every 100 epochs by default)
  python3 master.py --train --pgn_zip PGNs.zip --model model.npz --ngram 3 --epochs 2048 --checkpoint 100

  # Run as a bot + serve dashboard on port 10000 (reads token from env)
  export LICHESS_TOKEN="your_real_token_here"
  python3 master.py --bot --serve --port 10000 --model model.npz

Dependencies:
  pip install numpy python-chess berserk flask python-dotenv
"""
import argparse
import collections
import logging
import os
import random
import sys
import threading
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess
import chess.pgn
from flask import Flask, jsonify, Response

# optional convenience for local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# try import berserk but keep friendly error
try:
    import berserk
except Exception:
    berserk = None

# ---------------------------
# Configuration (change if needed)
# ---------------------------
LOG = logging.getLogger("iwcm-master")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MODEL_PATH = "model.npz"
LIVE_PGN_DIR = "live_games"
DEFAULT_PGN_DIR = "pgns"
DEFAULT_PGN_ZIP = "PGNs.zip"
DEFAULT_NGRAM = 3
DEFAULT_DISCOUNT = 0.75
SNAPSHOT_INTERVAL = 600  # seconds
HEARTBEAT_TIMEOUT = 180
MAX_RECONNECT_BACKOFF = 120
DEFAULT_TEMP = 0.6
DEFAULT_ARGMAX = False

# ---------------------------
# Lightweight Hybrid Model
# ---------------------------
class HybridModel:
    """Simple hybrid n-gram + FEN-count model.

    Stores:
      - context_counts: mapping from tuple(previous moves) -> Counter(next_move)
      - fen_counts: mapping from reduced FEN -> Counter(next_move)
      - unigram counts
    """
    def __init__(self, n: int = DEFAULT_NGRAM, discount: float = DEFAULT_DISCOUNT):
        self.n = int(n)
        self.discount = float(discount)
        self.context_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.fen_counts: Dict[str, Counter] = defaultdict(Counter)
        self.unigram: Counter = Counter()
        self.total_unigrams = 0
        self.games_trained = 0

    def feed_game(self, moves: List[str]):
        # update unigram & n-grams
        for i, mv in enumerate(moves):
            self.unigram[mv] += 1
            self.total_unigrams += 1
            for k in range(1, self.n + 1):
                if i - k < 0:
                    break
                ctx = tuple(moves[i - k : i])
                self.context_counts[ctx][mv] += 1
        # fen counts (before each move)
        board = chess.Board()
        for mv in moves:
            fen = self._reduce_fen(board.fen())
            self.fen_counts[fen][mv] += 1
            try:
                board.push_uci(mv)
            except Exception:
                break
        self.games_trained += 1

    def train_from_iterator(self, iterator):
        c = 0
        for moves in iterator:
            try:
                if moves:
                    self.feed_game(moves)
                    c += 1
            except Exception:
                LOG.exception("Error while feeding game")
        LOG.info("Trained on %d games (cumulative %d)", c, self.games_trained)
        return c

    def _reduce_fen(self, fen: str) -> str:
        parts = fen.split()
        # keep piece placement, active color, castling rights, en-passant
        if len(parts) >= 4:
            return " ".join(parts[:4])
        return fen

    def predict_probs(self, history: List[str], legal_moves: List[str], board: Optional[chess.Board] = None) -> Dict[str, float]:
        legal_moves = list(legal_moves)
        if not legal_moves:
            return {}
        # FEN heavy lookup has priority
        if board is not None:
            fen = self._reduce_fen(board.fen())
            if fen in self.fen_counts:
                cnts = self.fen_counts[fen]
                total = sum(cnts.values())
                if total > 0:
                    probs = {m: max(cnts.get(m, 0) - self.discount, 0) for m in legal_moves}
                    s = sum(probs.values())
                    if s > 0:
                        return {m: probs[m] / s for m in probs if probs[m] > 0}
        # n-gram backoff
        final = {m: 0.0 for m in legal_moves}
        remaining = 1.0
        for k in range(self.n, 0, -1):
            if len(history) < k:
                continue
            ctx = tuple(history[-k:])
            cnts = self.context_counts.get(ctx)
            if not cnts:
                continue
            N = sum(cnts.values())
            if N <= 0:
                continue
            unique = sum(1 for v in cnts.values() if v > 0)
            for m in legal_moves:
                c = cnts.get(m, 0)
                val = max(c - self.discount, 0.0) / N
                if val > 0:
                    final[m] += remaining * val
            backoff_mass = (self.discount * unique) / N if N > 0 else 1.0
            remaining *= backoff_mass
        # unigram fallback
        if remaining > 0:
            total_u = self.total_unigrams if self.total_unigrams > 0 else sum(self.unigram.values())
            if total_u > 0:
                for m in legal_moves:
                    final[m] += remaining * (self.unigram.get(m, 0) / total_u)
            else:
                for m in legal_moves:
                    final[m] += remaining * (1.0 / len(legal_moves))
        s = sum(final.values())
        if s <= 0:
            return {m: 1.0 / len(legal_moves) for m in legal_moves}
        out = {m: final[m] / s for m in final if final[m] > 0}
        for m in legal_moves:
            if m not in out:
                out[m] = 1e-12
        s2 = sum(out.values())
        return {m: out[m] / s2 for m in out}

    def sample(self, history: List[str], legal_moves: List[str], board: Optional[chess.Board] = None,
               temperature: float = 1.0, argmax: bool = False) -> Tuple[str, Dict[str, float]]:
        probs = self.predict_probs(history, legal_moves, board)
        if not probs:
            return random.choice(legal_moves), {}
        if argmax:
            mv = max(probs.items(), key=lambda kv: kv[1])[0]
            return mv, probs
        moves = list(probs.keys())
        p = np.array([probs[m] for m in moves], dtype=np.float64)
        if temperature != 1.0 and temperature > 0:
            logits = np.log(p + 1e-12) / temperature
            exps = np.exp(logits - logits.max())
            p = exps / exps.sum()
        p = p / p.sum()
        mv = np.random.choice(moves, p=p)
        return mv, dict(zip(moves, p.tolist()))

    def save(self, path: str):
        LOG.info("Saving model to %s", path)
        contexts = list(self.context_counts.keys())
        counts = [dict(self.context_counts[c]) for c in contexts]
        fen_keys = list(self.fen_counts.keys())
        fen_counts = [dict(self.fen_counts[f]) for f in fen_keys]
        np.savez_compressed(
            path,
            n=np.array([self.n]),
            discount=np.array([self.discount]),
            contexts=np.array(list(map(list, contexts)), dtype=object),
            counts=np.array(counts, dtype=object),
            fen_keys=np.array(fen_keys, dtype=object),
            fen_counts=np.array(fen_counts, dtype=object),
            unig_keys=np.array(list(self.unigram.keys()), dtype=object),
            unig_vals=np.array(list(self.unigram.values()), dtype=object),
            total_unigrams=np.array([self.total_unigrams]),
            games_trained=np.array([self.games_trained]),
        )

    @classmethod
    def load(cls, path: str):
        raw = None
        try:
            raw = np.load(path, allow_pickle=True)
        except Exception:
            LOG.debug("No model at %s", path)
            return None
        n = int(raw.get("n", np.array([DEFAULT_NGRAM])).tolist()[0])
        discount = float(raw.get("discount", np.array([DEFAULT_DISCOUNT])).tolist()[0])
        model = cls(n=n, discount=discount)
        contexts = raw.get("contexts", np.array([], dtype=object)).tolist() if "contexts" in raw.files else []
        counts = raw.get("counts", np.array([], dtype=object)).tolist() if "counts" in raw.files else []
        for ctx, cnt in zip(contexts, counts):
            model.context_counts[tuple(ctx)].update(cnt)
        fen_keys = raw.get("fen_keys", np.array([], dtype=object)).tolist() if "fen_keys" in raw.files else []
        fen_counts = raw.get("fen_counts", np.array([], dtype=object)).tolist() if "fen_counts" in raw.files else []
        for fk, fc in zip(fen_keys, fen_counts):
            model.fen_counts[fk].update(fc)
        if "unig_keys" in raw.files and "unig_vals" in raw.files:
            k = raw["unig_keys"].tolist()
            v = raw["unig_vals"].tolist()
            model.unigram.update(dict(zip(k, v)))
        model.total_unigrams = int(raw.get("total_unigrams", np.array([sum(model.unigram.values())])).tolist()[0])
        model.games_trained = int(raw.get("games_trained", np.array([0])).tolist()[0])
        LOG.info("Loaded model from %s: n=%d contexts=%d fens=%d unigrams=%d games=%d",
                 path, model.n, len(model.context_counts), len(model.fen_counts), len(model.unigram), model.games_trained)
        return model


# ---------------------------
# PGN helpers
# ---------------------------
def iter_pgn_file(filepath: str):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            yield g


def pgn_moves_from_game(game: chess.pgn.Game) -> List[str]:
    board = game.board()
    moves = []
    for mv in game.mainline_moves():
        try:
            moves.append(mv.uci())
            board.push(mv)
        except Exception:
            break
    return moves


def scan_pgn_dir(pgn_dir: str):
    p = Path(pgn_dir)
    if not p.exists():
        LOG.warning("PGN dir not found: %s", pgn_dir)
        return
    for child in sorted(p.iterdir()):
        if child.is_file() and child.suffix.lower() == ".pgn":
            for g in iter_pgn_file(str(child)):
                m = pgn_moves_from_game(g)
                if m:
                    yield m


def scan_pgn_zip(zip_path: str):
    if not os.path.exists(zip_path):
        LOG.warning("PGN zip not found: %s", zip_path)
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".pgn"):
                continue
            raw = zf.read(name).decode("utf-8", errors="replace")
            buf = StringIO(raw)
            while True:
                g = chess.pgn.read_game(buf)
                if g is None:
                    break
                m = pgn_moves_from_game(g)
                if m:
                    yield m


# ---------------------------
# Lichess Bot orchestration
# ---------------------------
class LichessBot:
    def __init__(self, token: str, model: HybridModel, temp: float = DEFAULT_TEMP, argmax: bool = DEFAULT_ARGMAX,
                 snapshot_interval: int = SNAPSHOT_INTERVAL):
        if berserk is None:
            raise RuntimeError("berserk library required: pip install berserk")
        self.token = token
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(session=self.session)
        self.model = model
        self.temp = temp
        self.argmax = argmax
        self.snapshot_interval = snapshot_interval
        self.active_games = {}  # game_id -> dict
        self.lock = threading.Lock()
        self._stop = threading.Event()
        os.makedirs(LIVE_PGN_DIR, exist_ok=True)

    def accept_challenge_if_supported(self, challenge: dict):
        cid = challenge.get("id")
        variant = challenge.get("variant", {}).get("key", "")
        try:
            if variant in ("standard", "fromPosition"):
                LOG.info("Accepting challenge %s variant=%s", cid, variant)
                self.client.bots.accept_challenge(cid)
            else:
                LOG.info("Declining non-standard challenge %s variant=%s", cid, variant)
                self.client.bots.decline_challenge(cid)
        except Exception:
            LOG.exception("Failed to handle challenge %s", cid)

    def handle_game(self, game_id: str, my_color: chess.Color):
        LOG.info("[game %s] handler start color=%s", game_id, "white" if my_color else "black")
        board = chess.Board()
        try:
            stream = self.client.bots.stream_game_state(game_id)
        except Exception:
            LOG.exception("Failed to stream game %s", game_id)
            return
        for event in stream:
            try:
                if event.get("type") in ("gameFull", "gameState"):
                    state = event.get("state", event)
                    moves_s = state.get("moves", "")
                    moves = moves_s.split() if moves_s else []
                    board = chess.Board()
                    for mv in moves:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            LOG.debug("Illegal move in stream %s: %s", game_id, mv)
                    with self.lock:
                        self.active_games.setdefault(game_id, {"moves": [], "last_thought": None, "white": None, "black": None, "result": None})
                        self.active_games[game_id]["moves"] = moves[:]
                    if not board.is_game_over() and board.turn == my_color:
                        legal = [m.uci() for m in board.legal_moves]
                        choice, probs = self.model.sample(moves, legal, board=board, temperature=self.temp, argmax=self.argmax)
                        LOG.info("[game %s] Playing %s", game_id, choice)
                        with self.lock:
                            self.active_games[game_id]["last_thought"] = {"choice": choice, "probs": probs}
                        try:
                            self.client.bots.make_move(game_id, choice)
                        except Exception:
                            LOG.exception("Failed to make move %s in game %s", choice, game_id)
                elif event.get("type") == "gameFinish":
                    LOG.info("[game %s] finished %s", game_id, event)
                    try:
                        pgn_text = self.client.games.export(game_id)
                        save_path = os.path.join(LIVE_PGN_DIR, f"{game_id}.pgn")
                        with open(save_path, "w", encoding="utf-8") as fh:
                            fh.write(pgn_text)
                        LOG.info("Exported finished PGN to %s", save_path)
                    except Exception:
                        LOG.exception("Failed to export pgn for %s", game_id)
                    with self.lock:
                        moves = self.active_games.get(game_id, {}).get("moves", [])[:]
                    if moves:
                        try:
                            self.model.feed_game(moves)
                            LOG.info("Online-trained with finished game %s (moves=%d)", game_id, len(moves))
                        except Exception:
                            LOG.exception("Failed to update model with finished game %s", game_id)
            except Exception:
                LOG.exception("Exception in game handler loop for %s", game_id)
            if self._stop.is_set():
                LOG.info("Stopping handler for %s due to stop flag", game_id)
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
                        typ = event.get("type")
                        LOG.debug("Incoming event: %s", typ)
                        if typ == "challenge":
                            self.accept_challenge_if_supported(event["challenge"])
                        elif typ == "gameStart":
                            gid = event["game"]["id"]
                            color = event["game"].get("color", "white")
                            my_color = chess.WHITE if color == "white" else chess.BLACK
                            t = threading.Thread(target=self.handle_game, args=(gid, my_color), daemon=True)
                            t.start()
                    except Exception:
                        LOG.exception("Error processing incoming event")
                LOG.warning("Incoming stream ended; reconnecting after %ds", backoff)
            except Exception as e:
                LOG.exception("Exception in incoming_loop: %s", e)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_RECONNECT_BACKOFF)

    def start(self):
        self._stop.clear()
        t = threading.Thread(target=self.incoming_loop, daemon=True)
        t.start()
        s = threading.Thread(target=self.snapshot_loop, daemon=True)
        s.start()

    def stop(self):
        self._stop.set()

    def snapshot_loop(self):
        while not self._stop.is_set():
            try:
                time.sleep(self.snapshot_interval)
                self.model.save(DEFAULT_MODEL_PATH)
                LOG.info("Snapshot saved")
            except Exception:
                LOG.exception("Snapshot error")


# ---------------------------
# Flask dashboard
# ---------------------------
APP = Flask(__name__)
GLOBAL_BOT: Optional[LichessBot] = None

DASH_HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>IWCM Bot Dashboard</title>
<style>body{font-family:system-ui;margin:18px}table{width:100%;border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px;text-align:left}</style>
</head>
<body>
<h1>IWCM Bot Dashboard</h1>
<p>Active games (auto-refresh every 3s)</p>
<div id="content"></div>
<script>
async function refresh(){const r=await fetch('/_games');const j=await r.json();let html='<table><tr><th>Game</th><th>White</th><th>Black</th><th>Moves</th><th>Last thought</th></tr>';for(const id of Object.keys(j)){const g=j[id];html+=`<tr><td>${id}</td><td>${g.white||''}</td><td>${g.black||''}</td><td style="font-family:monospace">${(g.moves||[]).join(' ')}</td><td style="font-family:monospace">${g.last_thought?JSON.stringify(g.last_thought):''}</td></tr>`;}html+='</table>';document.getElementById('content').innerHTML=html;}setInterval(refresh,3000);refresh();
</script>
</body>
</html>
"""

@APP.route('/')
def index():
    return DASH_HTML

@APP.route('/_games')
def api_games():
    if GLOBAL_BOT is None:
        return jsonify({})
    with GLOBAL_BOT.lock:
        return jsonify(GLOBAL_BOT.active_games)


# ---------------------------
# CLI and orchestration
# ---------------------------
def make_parser():
    p = argparse.ArgumentParser(description='IWCM master no-TF bot')
    p.add_argument('--train', action='store_true', help='Train from PGNs')
    p.add_argument('--pgn_dir', default=DEFAULT_PGN_DIR)
    p.add_argument('--pgn_zip', default=DEFAULT_PGN_ZIP)
    p.add_argument('--model', default=DEFAULT_MODEL_PATH)
    p.add_argument('--ngram', type=int, default=DEFAULT_NGRAM)
    p.add_argument('--discount', type=float, default=DEFAULT_DISCOUNT)
    p.add_argument('--bot', action='store_true')
    p.add_argument('--serve', action='store_true')
    p.add_argument('--port', type=int, default=10000)
    p.add_argument('--temp', type=float, default=DEFAULT_TEMP)
    p.add_argument('--argmax', action='store_true')
    p.add_argument('--snapshot', type=int, default=SNAPSHOT_INTERVAL)
    p.add_argument('--token', default=None, help='Optional: pass token directly (not recommended)')
    p.add_argument('--epochs', type=int, default=1, help='Number of full-pass epochs over the PGN dataset')
    p.add_argument('--checkpoint', type=int, default=100, help='Save model every N epochs')
    return p

def train_from_sources(model: HybridModel, pgn_dir: str = None, pgn_zip: str = None):
    total = 0
    if pgn_dir and os.path.exists(pgn_dir):
        total += model.train_from_iterator(scan_pgn_dir(pgn_dir))
    if pgn_zip and os.path.exists(pgn_zip):
        total += model.train_from_iterator(scan_pgn_zip(pgn_zip))
    return total

def main(argv=None):
    global GLOBAL_BOT
    args = make_parser().parse_args(argv)
    random.seed(42)
    np.random.seed(42)

    # load or create model
    model = None
    if os.path.exists(args.model):
        model = HybridModel.load(args.model)
    if model is None:
        model = HybridModel(n=args.ngram, discount=args.discount)

    if args.train:
        LOG.info('Training requested: epochs=%d', args.epochs)
        for epoch in range(args.epochs):
            LOG.info("Epoch %d/%d -- starting", epoch + 1, args.epochs)
            cnt = train_from_sources(model, pgn_dir=args.pgn_dir, pgn_zip=args.pgn_zip)
            LOG.info('Epoch %d done: trained on %d games (cumulative %d)', epoch + 1, cnt, model.games_trained)
            if (epoch + 1) % args.checkpoint == 0:
                try:
                    model.save(args.model)
                    LOG.info("Checkpoint saved at epoch %d to %s", epoch + 1, args.model)
                except Exception:
                    LOG.exception("Failed to save checkpoint at epoch %d", epoch + 1)
        # final save after all epochs
        try:
            model.save(args.model)
            LOG.info("Final model saved to %s", args.model)
        except Exception:
            LOG.exception("Failed to save final model")

    token = args.token or os.getenv('LICHESS_TOKEN')
    if args.bot and not token:
        LOG.error('Bot requested but no token provided. Set LICHESS_TOKEN env var or use --token')
        sys.exit(1)

    bot = None
    if args.bot:
        bot = LichessBot(token, model, temp=args.temp, argmax=args.argmax, snapshot_interval=args.snapshot)
        GLOBAL_BOT = bot
        bot.start()

    if args.serve:
        LOG.info('Starting Flask on port %d', args.port)
        try:
            APP.run(host='0.0.0.0', port=args.port, threaded=True)
        except Exception:
            LOG.exception('Flask crashed')
    else:
        if bot is not None:
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                LOG.info('Shutting down')
                bot.stop()
                model.save(args.model)
        else:
            LOG.info('Nothing to do; use --train or --bot')

if __name__ == '__main__':
    main()
