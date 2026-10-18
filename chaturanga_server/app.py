"""HTTP API around the rules engine, used by the web app in web/.

The server is stateless: every request carries the starting position and the moves played
so far, and the server replays them. This keeps undo, review and lessons trivial, and the
Python engine remains the only implementation of the rules.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chaturanga import DEFAULT_RULES, GameState, move_to_str, parse_move
from chaturanga.agents import AGENTS, LEVELS, NetworkUnavailable, ZeroAgent, make_agent, trained_evaluator, warm_up
from chaturanga.constants import PADATI
from chaturanga.tables import LAST_RANK

from .training_api import router as training_router

MAX_MOVES = 1000
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


class GameRequest(BaseModel):
    start_fen: Optional[str] = Field(None, description="Starting position; the standard setup if omitted")
    moves: List[str] = Field(default_factory=list, max_length=MAX_MOVES, description="Moves like 'e2e3'")


class AgentRequest(GameRequest):
    agent: str = "greedy"
    seed: Optional[int] = None
    level: Optional[str] = Field(None, description="For the trained agent: easy, medium, hard or impossible")


def replay(req: GameRequest):
    """Rebuild the game, returning (state, history). Raises HTTP 400 on bad input."""
    try:
        state = GameState.from_fen(req.start_fen) if req.start_fen else GameState.initial()
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(400, f"bad start_fen: {exc}") from exc
    history = []
    for i, text in enumerate(req.moves):
        if state.is_terminal():
            raise HTTPException(400, f"move {i + 1} ({text}) was played after the game ended")
        try:
            move = parse_move(text)
        except ValueError as exc:
            raise HTTPException(400, f"move {i + 1}: {exc}") from exc
        if not state.is_legal(move):
            raise HTTPException(400, f"move {i + 1} ({text}) is illegal in {state.to_fen()}")
        history.append(_history_entry(state, move))
        state = state.apply(move)
        history[-1]["check"] = state.in_check()
    return state, history


def _history_entry(state: GameState, move: int) -> dict:
    fr, to = divmod(move, 64)
    piece = state.board[fr]
    promoted = 0
    if abs(piece) == PADATI and to >> 3 == LAST_RANK[state.side]:
        after = state.apply(move)
        promoted = after.board[to]
    return {"move": move_to_str(move), "from": fr, "to": to, "piece": piece,
            "captured": state.board[to], "promotion": promoted, "check": False}


def game_view(state: GameState, history: list) -> dict:
    out = state.outcome()
    return {
        "fen": state.to_fen(),
        "board": state.board,
        "side": state.side,
        "ply": state.ply,
        "in_check": state.in_check(),
        "king_square": state.king_square(state.side),
        "legal_moves": [[m // 64, m % 64] for m in state.legal_moves()] if out is None else [],
        "outcome": None if out is None else {"winner": out.winner, "reason": out.reason},
        "no_progress": state.no_progress,
        "repetition": state.repetition_count(),
        "history": history,
    }


log = logging.getLogger("uvicorn.error")


def _warm_up_mcts() -> None:
    t = time.perf_counter()
    try:
        warm_up()
    except NetworkUnavailable as exc:  # the other agents still work; MCTS moves report this error
        log.error("%s", exc)
        return
    log.info("MCTS network ready (%.1f s)", time.perf_counter() - t)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load PyTorch and the network in the background: the server answers at once (Random and
    # Greedy need neither), and an early MCTS move simply waits for the load to finish.
    threading.Thread(target=_warm_up_mcts, name="mcts-warm-up", daemon=True).start()
    yield


app = FastAPI(title="ChaturangZero", version="0.2.0", lifespan=lifespan)
app.include_router(training_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    rules = DEFAULT_RULES
    return {
        "rules": {
            "setup": rules.setup, "promotion": rules.promotion, "stalemate": rules.stalemate,
            "repetition_limit": rules.repetition_limit, "no_progress_plies": rules.no_progress_plies,
            "max_plies": rules.max_plies,
        },
        "agents": [{"id": a.id, "name": a.name, "description": a.description} for a in AGENTS.values()],
        "levels": [{"id": lv.id, "name": lv.name, "simulations": lv.simulations, "description": lv.description}
                   for lv in LEVELS.values()],
        "trained": _trained_info(),
    }


def _trained_info():
    """The published trained network ({run, generation}), or None if there is none (yet)."""
    try:
        trained = trained_evaluator()
    except NetworkUnavailable:
        return None
    return trained[1] if trained else None


@app.post("/api/game")
def game(req: GameRequest):
    state, history = replay(req)
    return game_view(state, history)


@app.post("/api/agent-move")
def agent_move(req: AgentRequest):
    state, history = replay(req)
    if state.is_terminal():
        raise HTTPException(400, "the game is already over")
    try:
        agent = make_agent(req.agent, req.seed, req.level)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except NetworkUnavailable as exc:
        # 503: the server is fine, the MCTS network is not (yet) available. The app shows the
        # message with a Retry button.
        raise HTTPException(503, str(exc)) from exc
    move = agent.select_move(state)
    history.append(_history_entry(state, move))
    state = state.apply(move)
    history[-1]["check"] = state.in_check()
    out = {"move": move_to_str(move), "game": game_view(state, history)}
    if isinstance(agent, ZeroAgent):
        out["network"] = agent.network
    return out


# Serve the built web app (npm run build in web/) from the same server when it exists.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
