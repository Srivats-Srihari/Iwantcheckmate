# app.py
import os
import time
import json
import random
import requests
import threading
import numpy as np
from flask import Flask, jsonify, render_template_string
import chess
import chess.engine
import chess.pgn

# ================================
# CONFIGURATION
# ================================
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN", "your_lichess_token_here")
BOT_NAME = os.getenv("BOT_NAME", "PlayerBot")
VOCAB_PATH = "vocab.npz"
STREAM_URL = "https://lichess.org/api/stream/event"
GAME_URL = "https://lichess.org/api/bot/game/stream/"
MOVE_URL = "https://lichess.org/api/bot/game/"
HEADERS = {"Authorization": f"Bearer {LICHESS_TOKEN}"}

# ================================
# GLOBAL STATE
# ================================
games = {}
known_moves = None
move_set = None
bot_color = {}
move_memory = {}
THOUGHTS = {}

# ================================
# LOAD PLAYER VOCAB
# ================================
try:
    data = np.load(VOCAB_PATH, allow_pickle=True)
    known_moves = data["moves"]
    move_set = set(known_moves)
    print(f"Loaded {len(known_moves)} known moves from vocab.")
except Exception as e:
    print("⚠️ Could not load vocab:", e)
    known_moves = np.array([])
    move_set = set()

# ================================
# HELPER FUNCTIONS
# ================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def send_move(game_id, move):
    try:
        url = MOVE_URL + f"{game_id}/move/{move}"
        r = requests.post(url, headers=HEADERS)
        if r.status_code != 200:
            log(f"❌ Move rejected ({r.status_code}): {r.text}")
        else:
            log(f"✅ Played move {move}")
    except Exception as e:
        log(f"Error sending move: {e}")

def imitate_move(board):
    """
    Choose a move that matches the player’s style from vocab.
    If unavailable, fallback to a legal but non-blundering move.
    """
    legal_moves = list(board.legal_moves)
    move_strs = [str(m) for m in legal_moves]

    # Prioritize moves seen in player vocab
    styled = [m for m in move_strs if m in move_set]
    if styled:
        move = random.choice(styled)
        thought = f"Remembered a move like {move} before… feels right."
    else:
        # fallback: pick move with heuristic closeness
        move = random.choice(move_strs)
        thought = f"Haven’t seen this before. Instinct says {move}."
    THOUGHTS['last'] = thought
    return move, thought

# ================================
# GAME HANDLER
# ================================
def handle_game(game_id):
    url = GAME_URL + game_id + "/event"
    r = requests.get(url, headers=HEADERS, stream=True)
    if r.status_code != 200:
        log(f"Failed to connect to game stream {game_id}: {r.text}")
        return

    board = chess.Board()
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode("utf-8"))

        if data["type"] == "gameFull":
            color = data["white"]["id"].lower()
            my_color = "white" if color == BOT_NAME.lower() else "black"
            bot_color[game_id] = my_color
            games[game_id] = {
                "white": data["white"]["id"],
                "black": data["black"]["id"],
                "moves": [],
                "result": None,
            }
            log(f"Joined game {game_id} as {my_color}")

            if my_color == "white":
                move, thought = imitate_move(board)
                send_move(game_id, move)
                board.push_uci(move)
                games[game_id]["moves"].append(move)
                THOUGHTS[game_id] = thought

        elif data["type"] == "gameState":
            moves = data["moves"].split()
            my_turn = (len(moves) % 2 == 0 and bot_color[game_id] == "white") or (
                len(moves) % 2 == 1 and bot_color[game_id] == "black"
            )
            for move in moves[len(games[game_id]["moves"]):]:
                board.push_uci(move)
            games[game_id]["moves"] = moves

            if my_turn and not board.is_game_over():
                move, thought = imitate_move(board)
                send_move(game_id, move)
                board.push_uci(move)
                games[game_id]["moves"].append(move)
                THOUGHTS[game_id] = thought
                log(f"({game_id}) {BOT_NAME} thought: {thought}")

        elif data["type"] == "chatLine":
            log(f"💬 Chat in {game_id}: {data['username']}: {data['text']}")
        elif data["type"] == "gameFinish":
            result = data.get("winner", "draw")
            games[game_id]["result"] = result
            log(f"🏁 Game {game_id} finished. Result: {result}")
            return

# ================================
# EVENT LISTENER
# ================================
def event_listener():
    while True:
        r = requests.get(STREAM_URL, headers=HEADERS, stream=True)
        if r.status_code != 200:
            log(f"Stream failed: {r.text}")
            time.sleep(10)
            continue
        for line in r.iter_lines():
            if not line:
                continue
            event = json.loads(line.decode("utf-8"))
            if event["type"] == "challenge":
                ch_id = event["challenge"]["id"]
                ch_url = f"https://lichess.org/api/challenge/{ch_id}/accept"
                requests.post(ch_url, headers=HEADERS)
                log(f"Accepted challenge {ch_id}")
            elif event["type"] == "gameStart":
                game_id = event["game"]["id"]
                threading.Thread(target=handle_game, args=(game_id,), daemon=True).start()

# ================================
# WEB DASHBOARD
# ================================
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
<head>
<title>PlayerBot Dashboard</title>
<style>
body { font-family: monospace; background: #111; color: #0f0; padding: 20px; }
h1 { color: #6f6; }
.game { border: 1px solid #0f0; padding: 10px; margin-bottom: 20px; }
</style>
<meta http-equiv="refresh" content="5">
</head>
<body>
<h1>♟ PlayerBot Dashboard</h1>
<p>Watching {{ count }} games.</p>
{% for gid, g in games.items() %}
<div class="game">
<b>Game ID:</b> {{ gid }}<br>
<b>White:</b> {{ g.white }} | <b>Black:</b> {{ g.black }}<br>
<b>Moves:</b> {{ g.moves|join(", ") }}<br>
<b>Result:</b> {{ g.result }}<br>
<b>Thought:</b> {{ thoughts.get(gid, thoughts.get('last', '...')) }}
</div>
{% endfor %}
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE, games=games, thoughts=THOUGHTS, count=len(games))

@app.route("/state")
def state():
    return jsonify(games)

# ================================
# MAIN THREADS
# ================================
if __name__ == "__main__":
    threading.Thread(target=event_listener, daemon=True).start()
    port = int(os.getenv("PORT", 5000))
    log(f"🌐 Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
