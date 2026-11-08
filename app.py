import asyncio
import time
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from quant_engine import XAUQuantEngine
from data_feed import XAUDataFeed

app = FastAPI(title="XAU × QUANT Terminal", version="2.6.0")

feed = XAUDataFeed(history_bars=300)
engine = XAUQuantEngine(account_balance=5000.0, fixed_lot=0.01)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/export-snapshot")
async def export_snapshot():
    closes = np.array(feed.closes)
    opens = np.array(feed.opens)
    highs = np.array(feed.highs)
    lows = np.array(feed.lows)
    volumes = np.array(feed.volumes)

    knn = engine.knn_analog_matcher(closes)
    mc = engine.monte_carlo_ruin_cone(closes)
    bayes = engine.bayesian_kelly_update(closes, volumes, knn)
    flow = engine.order_flow_cvd_and_smc(opens, highs, lows, closes, volumes)

    return JSONResponse(
        content={
            "symbol": "XAU/USD (Spot/CFD)",
            "timestamp_utc": int(time.time()),
            "last_price": float(closes[-1]),
            "account_balance": engine.account_balance,
            "fixed_lot": engine.fixed_lot,
            "knn_matcher": knn,
            "monte_carlo": mc,
            "bayesian_kelly": bayes,
            "order_flow": flow,
        }
    )


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            t0 = time.perf_counter()
            tick_data = feed.step_tick()

            closes = np.array(feed.closes)
            opens = np.array(feed.opens)
            highs = np.array(feed.highs)
            lows = np.array(feed.lows)
            volumes = np.array(feed.volumes)

            knn = engine.knn_analog_matcher(closes)
            mc = engine.monte_carlo_ruin_cone(closes)
            bayes = engine.bayesian_kelly_update(closes, volumes, knn)
            flow = engine.order_flow_cvd_and_smc(opens, highs, lows, closes, volumes)

            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            payload = {
                "latency_ms": latency_ms,
                "market": tick_data,
                "knn": knn,
                "monte_carlo": mc,
                "bayesian": bayes,
                "order_flow": flow,
            }

            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.85)
    except WebSocketDisconnect:
        pass
