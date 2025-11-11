#!/usr/bin/env python3
# master_tf.py
"""
TensorFlow-based chess imitation trainer + Lichess bot (single-file)

Usage examples:
  # Build moves vocab and quick test training:
  python3 master_tf.py --train --pgn_zip PGNs.zip --model model.h5 --moves moves.npz --epochs 1 --batch 64

  # Run bot + tiny dashboard (ensure LICHESS_TOKEN env var is set)
  export LICHESS_TOKEN="your_token_here"
  python3 master_tf.py --bot --serve --model model.h5 --moves moves.npz --port 10000

Notes:
 - Assumes TensorFlow 2.x installed (`import tensorflow as tf`).
 - For heavy training use Colab / PC; Pi is OK for small fine-tune / inference.
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# core libs
import numpy as np
import chess
import chess.pgn

# TF
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    raise RuntimeError("TensorFlow import failed: %s" % e)

# berserk for Lichess
try:
    import berserk
except Exception:
    berserk = None

# optional stockfish
try:
    import chess.engine
    STOCKFISH_AVAILABLE = True
except Exception:
    chess.engine = None
    STOCKFISH_AVAILABLE = False

# flask for dashboard
try:
    from flask import Flask, jsonify
except Exception:
    Flask = None

# logging
LOG = logging.getLogger("iwcm-tf")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Move vocab generation
# -------------------------
def generate_all_uci_moves() -> List[str]:
    files = "abcdefgh"
    ranks = "12345678"
    promos = ["q", "r", "b", "n"]
    squares = [f + r for f in files for r in ranks]
    moves = []
    for a in squares:
        for b in squares:
            if a == b:
                continue
            moves.append(a + b)
            if b[1] in ("8", "1"):
                for p in promos:
                    moves.append(a + b + p)
    moves = sorted(set(moves))
    return moves

ALL_UCI = generate_all_uci_moves()
MOVE_TO_IDX = {m: i for i, m in enumerate(ALL_UCI)}
IDX_TO_MOVE = {i: m for m, i in MOVE_TO_IDX.items()}
OUTPUT_DIM = len(ALL_UCI)

# -------------------------
# Board -> tensor (channels_last)
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
    """
    Returns (8,8,13) float32 tensor: 12 piece planes + 1 side-to-move plane if requested.
    channels_last (H,W,C) for TF.
    """
    C = 12 + (1 if add_side_to_move else 0)
    arr = np.zeros((8, 8, C), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        pt = PIECE_TO_IDX[piece.piece_type]
        offset = 0 if piece.color == chess.WHITE else 6
        # convert to matrix coords: row 0 is rank 8 -> rank index reversed
        r = 7 - chess.square_rank(sq)
        f = chess.square_file(sq)
        arr[r, f, offset + pt] = 1.0
    if add_side_to_move:
        val = 1.0 if board.turn == chess.WHITE else 0.0
        arr[:, :, 12] = val
    return arr

# -------------------------
# PGN parsing -> dataset
# -------------------------
class PGNSampleBuilder:
    """
    Build dataset samples (board_tensor, move_idx) from PGN files (single .pgn files in a dir)
    or from a PGN zip. Saves a samples.npz if requested to avoid reparsing every run.
    """

    def __init__(self, pgn_dir: Optional[str]=None, pgn_zip: Optional[str]=None, moves_vocab: Dict[str,int]=MOVE_TO_IDX):
        if pgn_dir and pgn_zip:
            raise ValueError("Provide only one of pgn_dir or pgn_zip.")
        self.pgn_dir = pgn_dir
        self.pgn_zip = pgn_zip
        self.move_to_idx = moves_vocab
        self.samples = []  # tuple list of (board_tensor np.uint8/float32, int idx)

    def _parse_pgn_stream(self, fh):
        while True:
            g = chess.pgn.read_game(fh)
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

    def build(self):
        LOG.info("Building samples from pgn_dir=%s pgn_zip=%s", self.pgn_dir, self.pgn_zip)
        if self.pgn_dir and os.path.exists(self.pgn_dir):
            p = Path(self.pgn_dir)
            for path in sorted(p.rglob("*.pgn")):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        self._parse_pgn_stream(fh)
                except Exception:
                    LOG.exception("Failed to parse %s", path)
        if self.pgn_zip and os.path.exists(self.pgn_zip):
            with zipfile.ZipFile(self.pgn_zip, "r") as zf:
                for name in sorted(zf.namelist()):
                    if not name.lower().endswith(".pgn"):
                        continue
                    try:
                        raw = zf.read(name).decode("utf-8", errors="replace")
                        self._parse_pgn_stream(io.StringIO(raw))
                    except Exception:
                        LOG.exception("Failed to parse %s in zip", name)
        LOG.info("Built %d samples", len(self.samples))
        return self.samples

    def save_npz(self, out: str):
        if not self.samples:
            raise RuntimeError("No samples to save.")
        boards = np.stack([s[0] for s in self.samples], axis=0).astype(np.float32)
        targets = np.array([s[1] for s in self.samples], dtype=np.int32)
        np.savez_compressed(out, boards=boards, targets=targets)
        LOG.info("Saved %d samples to %s", boards.shape[0], out)

# -------------------------
# TF model (conv) builder
# -------------------------
def build_model(input_shape=(8,8,13), channels=64, output_dim=OUTPUT_DIM):
    inputs = keras.Input(shape=input_shape, name="board")
    x = layers.Conv2D(channels, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(channels*2, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(channels*2, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_dim, activation=None, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="conv_imitator")
    return model

# -------------------------
# TF training helper
# -------------------------
def make_tf_dataset_from_npz(npz_path, batch=64, shuffle=True, buffer=2048, val_split=0.05):
    d = np.load(npz_path, allow_pickle=True)
    boards = d["boards"]
    targets = d["targets"]
    n = len(targets)
    LOG.info("Loaded npz with %d samples", n)
    # shuffle indices
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1.0 - val_split))
    train_idx = idx[:split]
    val_idx = idx[split:]
    def gen(idx_list):
        for i in idx_list:
            yield boards[i], targets[i]
    out_train = tf.data.Dataset.from_generator(lambda: gen(train_idx), output_signature=(
        tf.TensorSpec(shape=boards.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ))
    out_val = tf.data.Dataset.from_generator(lambda: gen(val_idx), output_signature=(
        tf.TensorSpec(shape=boards.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ))
    if shuffle:
        out_train = out_train.shuffle(buffer)
    out_train = out_train.batch(batch).prefetch(tf.data.AUTOTUNE)
    out_val = out_val.batch(batch).prefetch(tf.data.AUTOTUNE)
    return out_train, out_val

# -------------------------
# Inference utils: mask illegal moves
# -------------------------
def predict_move_from_model_tf(model: keras.Model, board: chess.Board, temperature: float=1.0, argmax: bool=False):
    bt = board_to_tensor(board)
    bat = np.expand_dims(bt, axis=0).astype(np.float32)
    logits = model.predict(bat, verbose=0)[0]  # shape (OUTPUT_DIM,)
    legal = [m.uci() for m in board.legal_moves]
    mask = np.zeros_like(logits, dtype=bool)
    for m in legal:
        if m in MOVE_TO_IDX:
            mask[MOVE_TO_IDX[m]] = True
    if not mask.any():
        return None, {}
    big_neg = -1e9
    masked = np.where(mask, logits, big_neg)
    if temperature != 1.0 and temperature > 0:
        scaled = masked / float(temperature)
    else:
        scaled = masked
    scaled = scaled - np.max(scaled)
    exps = np.exp(scaled)
    exps = exps * mask
    probs = exps / (np.sum(exps) + 1e-12)
    if argmax:
        idx = int(np.argmax(probs))
    else:
        idx = int(np.random.choice(len(probs), p=probs))
    mv = IDX_TO_MOVE.get(idx, None)
    probs_dict = {IDX_TO_MOVE[i]: float(probs[i]) for i in range(len(probs)) if mask[i] and probs[i] > 0}
    return mv, probs_dict

# -------------------------
# Lichess Bot Class (TF)
# -------------------------
class TFTorchLichessBot:
    def __init__(self, token: str, model: keras.Model, moves_path: str, temp: float=0.7, argmax: bool=False, stockfish_path: Optional[str]=None):
        if berserk is None:
            raise RuntimeError("berserk library is required: pip install berserk")
        self.token = token
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(session=self.session)
        self.model = model
        self.temp = temp
        self.argmax = argmax
        self.active_games = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        if stockfish_path and STOCKFISH_AVAILABLE:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            except Exception:
                LOG.exception("Failed to start stockfish")
                self.engine = None
        else:
            self.engine = None

    def accept_challenge(self, challenge):
        cid = challenge.get("id")
        variant = challenge.get("variant", {}).get("key", "")
        # Accept standard and fromPosition only
        if variant in ("standard", "fromPosition"):
            try:
                self.client.bots.accept_challenge(cid)
                LOG.info("Accepted challenge %s", cid)
            except Exception:
                LOG.exception("Failed to accept %s", cid)
        else:
            try:
                self.client.bots.decline_challenge(cid)
                LOG.info("Declined challenge %s", cid)
            except Exception:
                LOG.exception("Failed to decline %s", cid)

    def handle_game(self, game_id: str, my_color: chess.Color):
        LOG.info("Start handler %s color=%s", game_id, 'white' if my_color else 'black')
        try:
            stream = self.client.bots.stream_game_state(game_id)
        except Exception:
            LOG.exception("Failed to open game stream %s", game_id)
            return
        for event in stream:
            try:
                if event.get("type") in ("gameFull", "gameState"):
                    state = event.get("state", event)
                    moves_s = state.get("moves", "")
                    moves = moves_s.split() if moves_s else []
                    board = chess.Board()
                    for m in moves:
                        try:
                            board.push_uci(m)
                        except Exception:
                            pass
                    with self.lock:
                        self.active_games.setdefault(game_id, {"moves": [], "last_thought": None, "white": None, "black": None, "result": None})
                        self.active_games[game_id]["moves"] = moves[:]
                    if not board.is_game_over() and board.turn == my_color:
                        # model inference
                        choice, probs = predict_move_from_model_tf(self.model, board, temperature=self.temp, argmax=self.argmax)
                        if choice is None:
                            # fallback
                            if self.engine:
                                try:
                                    r = self.engine.play(board, chess.engine.Limit(time=0.05))
                                    choice = r.move.uci()
                                except Exception:
                                    choice = None
                            if choice is None:
                                legal = list(board.legal_moves)
                                # simple heuristic: capture preferred else center
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
                        LOG.info("Game %s play %s", game_id, choice)
                        with self.lock:
                            self.active_games[game_id]["last_thought"] = {"choice": choice, "probs": probs}
                        try:
                            self.client.bots.make_move(game_id, choice)
                        except Exception:
                            LOG.exception("Failed to send move %s for game %s", choice, game_id)
                elif event.get("type") == "gameFinish":
                    LOG.info("Game %s finished: %s", game_id, event)
                    # try to export pgn
                    try:
                        pgn_text = self.client.games.export(game_id)
                        outp = Path("live_games")
                        outp.mkdir(exist_ok=True)
                        with open(outp / f"{game_id}.pgn", "w", encoding="utf-8") as fh:
                            fh.write(pgn_text)
                        LOG.info("Saved finished PGN %s", game_id)
                    except Exception:
                        LOG.exception("Failed to export game %s", game_id)
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
                        typ = event.get("type")
                        LOG.debug("Incoming event %s", typ)
                        if typ == "challenge":
                            self.accept_challenge(event["challenge"])
                        elif typ == "gameStart":
                            gid = event["game"]["id"]
                            color = event["game"].get("color", "white")
                            my_color = chess.WHITE if color == "white" else chess.BLACK
                            t = threading.Thread(target=self.handle_game, args=(gid, my_color), daemon=True)
                            t.start()
                    except Exception:
                        LOG.exception("Error processing incoming event")
                LOG.warning("Incoming stream closed; reconnecting in %ds", backoff)
            except Exception:
                LOG.exception("Exception in incoming loop")
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
# CLI / orchestration
# -------------------------
APP = Flask(__name__) if Flask else None
GLOBAL_BOT = None

if APP:
    @APP.route("/")
    def index():
        return "IWCM TensorFlow Bot Running"

    @APP.route("/_games")
    def api_games():
        if GLOBAL_BOT is None:
            return jsonify({})
        with GLOBAL_BOT.lock:
            return jsonify(GLOBAL_BOT.active_games)

def train_main(args):
    # 1) build moves.npz if requested
    if args.moves and not os.path.exists(args.moves):
        LOG.info("Saving moves vocab to %s", args.moves)
        np.savez_compressed(args.moves, moves=np.array(ALL_UCI, dtype=object))

    # 2) build samples or load existing samples npz
    samples_npz = args.samples_npz or "samples.npz"
    if args.pgn_dir or args.pgn_zip:
        builder = PGNSampleBuilder(pgn_dir=args.pgn_dir, pgn_zip=args.pgn_zip)
        builder.build()
        if len(builder.samples) == 0:
            LOG.error("No samples built from PGNs.")
            return
        # Save as samples.npz for faster reuse
        LOG.info("Saving samples to %s", samples_npz)
        boards = np.stack([s[0] for s in builder.samples], axis=0).astype(np.float32)
        targets = np.array([s[1] for s in builder.samples], dtype=np.int32)
        np.savez_compressed(samples_npz, boards=boards, targets=targets)
    elif not os.path.exists(samples_npz):
        LOG.error("No PGNs provided and no samples npz found.")
        return

    # 3) make tf datasets
    train_ds, val_ds = make_tf_dataset_from_npz(samples_npz, batch=args.batch, shuffle=True, val_split=args.val_split)

    # 4) build or load model
    model = build_model(input_shape=(8,8,13), channels=args.channels)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])
    start_epoch = 0
    if args.model and os.path.exists(args.model):
        try:
            model = keras.models.load_model(args.model)
            LOG.info("Loaded model from %s", args.model)
        except Exception:
            LOG.exception("Failed to load model; will train from scratch.")

    # 5) callbacks
    callbacks = []
    if args.model:
        ckpt = keras.callbacks.ModelCheckpoint(args.model, save_best_only=False, save_weights_only=False, verbose=1)
        callbacks.append(ckpt)
    if args.early_stop:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))

    # 6) train
    LOG.info("Training: epochs=%d batch=%d samples ~ %d", args.epochs, args.batch, sum(1 for _ in train_ds.unbatch()))
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    if args.model:
        try:
            model.save(args.model)
            LOG.info("Saved final model to %s", args.model)
        except Exception:
            LOG.exception("Failed to save final model")

def bot_main(args):
    if not args.model or not os.path.exists(args.model):
        LOG.error("Model file required for bot mode.")
        return
    LOG.info("Loading model from %s", args.model)
    model = keras.models.load_model(args.model)
    token = args.token or os.getenv("LICHESS_TOKEN")
    if not token:
        LOG.error("No Lichess token provided (env LICHESS_TOKEN or --token).")
        return
    bot = TFTorchLichessBot(token, model, moves_path=args.moves, temp=args.temp, argmax=args.argmax, stockfish_path=args.stockfish)
    global GLOBAL_BOT
    GLOBAL_BOT = bot
    bot.start()
    if args.serve and APP:
        APP.run(host="0.0.0.0", port=args.port, threaded=True)
    else:
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            bot.stop()

def make_parser():
    p = argparse.ArgumentParser(description="IWCM TensorFlow master")
    p.add_argument("--train", action="store_true")
    p.add_argument("--pgn_dir", default=None)
    p.add_argument("--pgn_zip", default=None)
    p.add_argument("--samples_npz", default=None)
    p.add_argument("--model", default="model.h5")
    p.add_argument("--moves", default="moves.npz")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--bot", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=10000)
    p.add_argument("--token", default=None)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--argmax", action="store_true")
    p.add_argument("--stockfish", default=None)
    return p

def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.train:
        train_main(args)
    elif args.bot:
        bot_main(args)
    else:
        print("Nothing to do. Use --train or --bot")

if __name__ == "__main__":
    main()
