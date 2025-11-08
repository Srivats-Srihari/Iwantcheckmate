#!/usr/bin/env python3
"""
master.py

No-TensorFlow, high-fidelity imitation bot for Lichess.

Features:
- Uses hybrid model (FEN -> move counts + N-gram context counts + unigram)
- Incremental online training from PGNs and live completed games
- Lichess bot integration using berserk (accepts standard/fromPosition challenges)
- Simple Flask dashboard showing active games & last 'thoughts'
- Periodic snapshots to disk; lightweight format (NPZ and PGNs)
- Backoff smoothing (absolute-discount style) and temperature/argmax options
- Robust to 429/401 errors and reconnects

Important: This file contains your token as provided. Keep it secret. Do not share publicly.

Dependencies:
  pip install numpy python-chess berserk flask

Run examples:
  # train from PGN directory:
  python3 master.py --train --pgn_dir pgns --model model.npz --ngram 3

  # run bot + serve dashboard (fills LICHESS_TOKEN from token variable below):
  python3 master.py --bot --serve --model model.npz --port 10000 --temp 0.6

"""

import argparse
import collections
import json
import logging
import os
import random
import signal
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
from flask import Flask, jsonify, render_template_string

# Attempt to import berserk (Lichess API client)
try:
    import berserk
except Exception:
    berserk = None

# --------------------- USER CONFIGURATION (EDIT CAREFULLY) -----------------
# Your Lichess bot token (you provided it). Keep it secret. Consider setting
# an environment variable instead of hardcoding in production.
LICHESS_TOKEN = "lip_zu8iGjLJFwwPKmKwIXTk"

# Filenames
DEFAULT_MODEL_PATH = "model.npz"   # model snapshot
LIVE_PGN_DIR = "live_games"        # finished games exported from Lichess
PGN_DIR = "pgns"                   # local training PGNs folder (optional)
VOCAB_PATH = "vocab.npz"           # optional vocab file

# Performance / safety tuning
DEFAULT_NGRAM = 3
DEFAULT_DISCOUNT = 0.75
SNAPSHOT_INTERVAL = 600  # seconds
HEARTBEAT_TIMEOUT = 180  # if no events in this many seconds, reconnect
MAX_RECONNECT_BACKOFF = 120

# Bot sampling defaults
DEFAULT_TEMP = 0.6
DEFAULT_ARGMAX = False

# Flask config
FLASK_DEBUG = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("iwcm-master")

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_load_npz(path: str):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return data
    except Exception:
        LOG.exception("Failed to load npz: %s", path)
        return None


# ----------------------------------------------------------------------------
# PGN parsing helpers
# ----------------------------------------------------------------------------

def iter_pgn_file(filepath: str):
    """Yield chess.pgn.Game objects from a file with possibly many PGNs."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        while True:
            try:
                g = chess.pgn.read_game(f)
            except Exception:
                LOG.exception("Error reading PGN in %s", filepath)
                break
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
            # skip ill-formed move
            LOG.debug("Skipping illegal move in PGN: %s", mv)
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
            LOG.info("Reading PGN %s from zip", name)
            raw = zf.read(name).decode("utf-8", errors="replace")
            buf = StringIO(raw)
            while True:
                g = chess.pgn.read_game(buf)
                if g is None:
                    break
                m = pgn_moves_from_game(g)
                if m:
                    yield m


# ----------------------------------------------------------------------------
# Hybrid imitation model (FEN counts + n-gram counts + unigram)
# ----------------------------------------------------------------------------

class HybridModel:
    def __init__(self, n: int = DEFAULT_NGRAM, discount: float = DEFAULT_DISCOUNT):
        self.n = int(n)
        self.discount = float(discount)
        self.context_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.fen_counts: Dict[str, Counter] = defaultdict(Counter)
        self.unigram: Counter = Counter()
        self.total_unigrams = 0
        self.games_trained = 0

    # -- training --
    def feed_game(self, moves: List[str]):
        # update unigram and context counts
        for i, mv in enumerate(moves):
            self.unigram[mv] += 1
            self.total_unigrams += 1
            for k in range(1, self.n + 1):
                if i - k < 0:
                    break
                ctx = tuple(moves[i - k : i])
                self.context_counts[ctx][mv] += 1
        # fen counts
        board = chess.Board()
        for mv in moves:
            fen = self._reduce_fen(board.fen())
            self.fen_counts[fen][mv] += 1
            try:
                board.push_uci(mv)
            except Exception:
                break
        self.games_trained += 1

    def train_from_iterator(self, it):
        c = 0
        for moves in it:
            try:
                self.feed_game(moves)
                c += 1
            except Exception:
                LOG.exception("Error feeding game")
        LOG.info("Trained on %d games (total games_trained=%d)", c, self.games_trained)
        return c

    # -- save/load --
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
        raw = safe_load_npz(path)
        if raw is None:
            return None
        n = int(raw.get("n", np.array([DEFAULT_NGRAM])).tolist()[0])
        discount = float(raw.get("discount", np.array([DEFAULT_DISCOUNT])).tolist()[0])
        model = cls(n=n, discount=discount)
        contexts = raw["contexts"].tolist() if "contexts" in raw.files else []
        counts = raw["counts"].tolist() if "counts" in raw.files else []
        for ctx, cnt in zip(contexts, counts):
            model.context_counts[tuple(ctx)].update(cnt)
        fen_keys = raw.get("fen_keys", np.array([], dtype=object)).tolist() if "fen_keys" in raw.files else []
        fen_counts = raw.get("fen_counts", np.array([], dtype=object)).tolist() if "fen_counts" in raw.files else []
        for fk, fc in zip(fen_keys, fen_counts):
            model.fen_counts[fk].update(fc)
        unig_keys = raw.get("unig_keys", raw.get("unigram_keys", None))
        if unig_keys is None:
            # attempt fallback names
            unig_keys = raw[raw.files[0]] if raw.files else []
        # load unigram explicitly
        if "unig_keys" in raw.files or "unigram_keys" in raw.files:
            k = raw.get("unig_keys", raw.get("unigram_keys"))
            v = raw.get("unig_vals", raw.get("unigram_vals"))
            if k is not None and v is not None:
                model.unigram.update(dict(zip(k.tolist(), v.tolist())))
        elif "unig_keys" not in raw.files and "unigram_keys" not in raw.files:
            # fallback: if there is no unigram, rebuild total_unigrams
            model.total_unigrams = sum(model.unigram.values())
        model.total_unigrams = int(raw.get("total_unigrams", np.array([sum(model.unigram.values())])).tolist()[0])
        model.games_trained = int(raw.get("games_trained", np.array([0])).tolist()[0])
        LOG.info("Loaded model %s: n=%d contexts=%d fen=%d unigrams=%d games=%d",
                 path, model.n, len(model.context_counts), len(model.fen_counts), len(model.unigram), model.games_trained)
        return model

    # -- prediction & sampling --
    def _reduce_fen(self, fen: str) -> str:
        parts = fen.split()
        if len(parts) >= 4:
            return " ".join(parts[:4])
        return fen

    def predict_probs(self, history: List[str], legal_moves: List[str], board: Optional[chess.Board] = None) -> Dict[str, float]:
        legal_moves = list(legal_moves)
        if not legal_moves:
            return {}
        # FEN lookup first
        if board is not None:
            fen = self._reduce_fen(board.fen())
            if fen in self.fen_counts:
                cnts = self.fen_counts[fen]
                total = sum(cnts.values())
                if total > 0:
                    probs = {m: max(cnts.get(m, 0) - self.discount, 0) / total for m in legal_moves}
                    s = sum(probs.values())
                    if s > 0:
                        return {m: probs[m] / s for m in probs if probs[m] > 0}
        # N-gram backoff
        final = {m: 0.0 for m in legal_moves}
        remaining = 1.0
        for k in range(self.n, 0, -1):
            if len(history) < k:
                continue
            ctx = tuple(history[-k:])
            cnts = self.context_counts.get(ctx, None)
            if not cnts:
                continue
            N = sum(cnts.values())
            if N <= 0:
                continue
            unique = sum(1 for v in cnts.values() if v > 0)
            # discounted probs
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
        # filter zero and normalize
        out = {m: final[m] / s for m in final if final[m] > 0}
        # ensure all legal moves present (with tiny eps)
        for m in legal_moves:
            if m not in out:
                out[m] = 1e-12
        # renormalize
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


# ----------------------------------------------------------------------------
# Lichess bot orchestration
# ----------------------------------------------------------------------------

class LichessBot:
    def __init__(self, token: str, model: HybridModel, temp: float = DEFAULT_TEMP, argmax: bool = DEFAULT_ARGMAX,
                 snapshot_interval: int = SNAPSHOT_INTERVAL):
        if berserk is None:
            raise RuntimeError("berserk library not installed. run: pip install berserk")
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
        ensure_dir(LIVE_PGN_DIR)

    def accept_or_decline(self, challenge: dict):
        cid = challenge.get("id")
        variant = challenge.get("variant", {}).get("key", "")
        if variant in ("standard", "fromPosition"):
            try:
                LOG.info("Accepting challenge %s variant=%s", cid, variant)
                self.client.bots.accept_challenge(cid)
            except Exception:
                LOG.exception("Failed to accept %s", cid)
        else:
            try:
                LOG.info("Declining challenge %s variant=%s", cid, variant)
                self.client.bots.decline_challenge(cid)
            except Exception:
                LOG.exception("Failed to decline %s", cid)

    def handle_game(self, game_id: str, my_color: chess.Color):
        LOG.info("[game %s] handler start color=%s", game_id, "white" if my_color else "black")
        board = chess.Board()
        try:
            stream = self.client.bots.stream_game_state(game_id)
        except Exception:
            LOG.exception("Failed to stream game state %s", game_id)
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
                        self.active_games[game_id]["moves"] = moves[:]  # copy
                    if not board.is_game_over() and board.turn == my_color:
                        legal = [m.uci() for m in board.legal_moves]
                        choice, probs = self.model.sample(moves, legal, board=board, temperature=self.temp, argmax=self.argmax)
                        LOG.info("[game %s] Playing %s (legal %d)", game_id, choice, len(legal))
                        with self.lock:
                            self.active_games[game_id]["last_thought"] = {"choice": choice, "probs": probs}
                        try:
                            self.client.bots.make_move(game_id, choice)
                        except Exception:
                            LOG.exception("Failed to make move %s in game %s", choice, game_id)
                    else:
                        LOG.debug("[game %s] Not our turn or game over", game_id)
                elif event.get("type") == "gameFinish":
                    LOG.info("[game %s] finished %s", game_id, event)
                    # try export PGN
                    try:
                        pgn_text = self.client.games.export(game_id)
                        save_path = os.path.join(LIVE_PGN_DIR, f"{game_id}.pgn")
                        with open(save_path, "w", encoding="utf-8") as fh:
                            fh.write(pgn_text)
                        LOG.info("Exported finished PGN to %s", save_path)
                    except Exception:
                        LOG.exception("Failed to export pgn for %s", game_id)
                    # update model with moves we kept
                    with self.lock:
                        moves = self.active_games.get(game_id, {}).get("moves", [])[:]
                    if moves:
                        try:
                            self.model.feed_game(moves)
                            LOG.info("Online trained with finished game %s (moves=%d)", game_id, len(moves))
                        except Exception:
                            LOG.exception("Failed to update model with finished game %s", game_id)
                # store other metadata if present
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
                        LOG.debug("Incoming event: %s", event.get("type"))
                        if event.get("type") == "challenge":
                            self.accept_or_decline(event["challenge"])
                        elif event.get("type") == "gameStart":
                            gid = event["game"]["id"]
                            color = event["game"].get("color", "white")
                            my_color = chess.WHITE if color == "white" else chess.BLACK
                            t = threading.Thread(target=self.handle_game, args=(gid, my_color), daemon=True)
                            t.start()
                    except Exception:
                        LOG.exception("Error processing incoming event")
                # stream ended; reconnect
                LOG.warning("Incoming events stream ended; reconnecting after %ds", backoff)
            except berserk.exceptions.ResponseError as e:
                LOG.exception("Berserk response error: %s", e)
                # handle rate-limiting / unauthorized
                try:
                    status = getattr(e, 'status_code', None)
                    LOG.info("ResponseError status: %s", status)
                except Exception:
                    pass
            except Exception:
                LOG.exception("Unexpected exception in incoming_loop")
            # progressive backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_RECONNECT_BACKOFF)

    def start(self):
        self._stop.clear()
        t = threading.Thread(target=self.incoming_loop, daemon=True)
        t.start()
        # snapshot thread
        s = threading.Thread(target=self.snapshot_loop, daemon=True)
        s.start()

    def stop(self):
        self._stop.set()

    def snapshot_loop(self):
        while not self._stop.is_set():
            try:
                time.sleep(self.snapshot_interval)
                self.model.save(DEFAULT_MODEL_PATH)
                LOG.info("Snapshot saved to %s", DEFAULT_MODEL_PATH)
            except Exception:
                LOG.exception("Snapshot error")


# ----------------------------------------------------------------------------
# Flask dashboard
# ----------------------------------------------------------------------------

APP = Flask(__name__)
GLOBAL_BOT: Optional[LichessBot] = None

DASH_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Bot Dashboard</title>
  <style>body{font-family:system-ui;margin:18px}table{width:100%;border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px;text-align:left}</style>
</head>
<body>
  <h1>IWantCheckmate — Bot Dashboard</h1>
  <p>Active games (auto refresh every 3s)</p>
  <div id="content"></div>
  <script>
    async function refresh(){
      const r = await fetch('/_games');
      const j = await r.json();
      let html = '<table><tr><th>Game</th><th>White</th><th>Black</th><th>Moves</th><th>Last thought</th></tr>';
      for(const id of Object.keys(j)){
        const g = j[id];
        html += `<tr><td>${id}</td><td>${g.white||''}</td><td>${g.black||''}</td><td style="font-family:monospace">${(g.moves||[]).join(' ')}</td><td style="font-family:monospace">${g.last_thought?JSON.stringify(g.last_thought):''}</td></tr>`;
      }
      html += '</table>';
      document.getElementById('content').innerHTML = html;
    }
    setInterval(refresh,3000); refresh();
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


# ----------------------------------------------------------------------------
# CLI & main orchestration
# ----------------------------------------------------------------------------


def make_parser():
    p = argparse.ArgumentParser(description="IWantCheckmate master no-TF bot")
    p.add_argument('--train', action='store_true', help='Train model from PGNs')
    p.add_argument('--pgn_dir', default=PGN_DIR, help='PGN directory for training')
    p.add_argument('--pgn_zip', default='', help='PGN zip for training')
    p.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Model save/load path (.npz)')
    p.add_argument('--ngram', type=int, default=DEFAULT_NGRAM)
    p.add_argument('--discount', type=float, default=DEFAULT_DISCOUNT)
    p.add_argument('--bot', action='store_true', help='Run as lichess bot')
    p.add_argument('--serve', action='store_true', help='Start Flask dashboard')
    p.add_argument('--port', type=int, default=10000)
    p.add_argument('--temp', type=float, default=DEFAULT_TEMP)
    p.add_argument('--argmax', action='store_true', help='Play deterministically')
    p.add_argument('--snapshot', type=int, default=SNAPSHOT_INTERVAL)
    return p


def main(argv=None):
    global GLOBAL_BOT
    args = make_parser().parse_args(argv)
    random.seed(1234)
    np.random.seed(1234)

    model = None
    # try load model
    if os.path.exists(args.model):
        try:
            model = HybridModel.load(args.model)
        except Exception:
            LOG.exception("Failed to load model %s", args.model)
            model = None
    # construct if missing
    if model is None:
        model = HybridModel(n=args.ngram, discount=args.discount)

    # TRAIN
    if args.train:
        LOG.info("Training model: n=%d discount=%.3f", args.ngram, args.discount)
        cnt = 0
        if args.pgn_dir and os.path.exists(args.pgn_dir):
            cnt = model.train_from_iterator(scan_pgn_dir(args.pgn_dir))
        elif args.pgn_zip and os.path.exists(args.pgn_zip):
            cnt = model.train_from_iterator(scan_pgn_zip(args.pgn_zip))
        else:
            LOG.error("No PGNs found. Provide --pgn_dir or --pgn_zip")
            sys.exit(1)
        model.save(args.model)
        LOG.info("Training finished, saved model to %s", args.model)

    bot = None
    # BOT
    if args.bot:
        token = os.environ.get('LICHESS_TOKEN') or LICHESS_TOKEN
        if not token:
            LOG.error('No LICHESS_TOKEN set; provide via env or edit file')
            sys.exit(1)
        bot = LichessBot(token, model, temp=args.temp, argmax=args.argmax, snapshot_interval=args.snapshot)
        GLOBAL_BOT = bot
        bot.start()

    # FLASK
    if args.serve:
        # run flask in main thread
        LOG.info('Starting Flask dashboard on port %d', args.port)
        try:
            APP.run(host='0.0.0.0', port=args.port, debug=FLASK_DEBUG, threaded=True)
        except Exception:
            LOG.exception('Flask failed')
    else:
        if bot:
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                LOG.info('Keyboard interrupt, shutting down')
                bot.stop()
                model.save(args.model)
        else:
            LOG.info('Nothing to do; use --train or --bot')


if __name__ == '__main__':
    main()
