# FastAPI service for the fraud model trained in notebooks/01_fraud_detection_model.ipynb
#
# Run from the project folder:
#   uvicorn api.fraud_api:app --port 8030
# Docs page: http://127.0.0.1:8030/docs

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "models", "fraud_model.pkl")
HISTORY_PATH = os.environ.get("FRM_HISTORY", r"E:\Fraud_Risk_Data\history_until_day150.parquet")

H1 = 3600
H24 = 86400
D7 = 7 * 86400


class Payment(BaseModel):
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D10: Optional[float] = None
    D15: Optional[float] = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None


def new_customer():
    return {"recent": deque(), "n": 0, "total": 0.0, "last_t": None, "devices": set(), "emails": set()}


def new_card():
    return {"recent": deque(), "n": 0, "total": 0.0, "addrs": set()}


state = {}


def text(value):
    return "nan" if value is None else str(value)


def make_keys(p):
    day = p["TransactionDT"] // 86400
    start = None if p["D1"] is None else float(day - p["D1"])
    uid = str(p["card1"]) + "_" + text(p["addr1"]) + "_" + text(start)
    card_key = str(p["card1"])
    for col in ["card2", "card3", "card4", "card5", "card6"]:
        card_key = card_key + "_" + text(p[col])
    return uid, card_key


def window_features(p, uid, card_key):
    t = float(p["TransactionDT"])
    amt = float(p["TransactionAmt"])
    device, email, addr = p["DeviceInfo"], p["P_emaildomain"], p["addr1"]
    u = state["customers"][uid]
    c = state["cards"][card_key]

    while u["recent"] and u["recent"][0][0] < t - D7:
        u["recent"].popleft()
    while c["recent"] and c["recent"][0][0] < t - D7:
        c["recent"].popleft()

    cnt_1h, cnt_24h, cnt_7d, amt_24h = 0, 0, 0, 0.0
    products = set()
    for old_t, old_amt, product in reversed(u["recent"]):
        gap = t - old_t
        cnt_7d += 1
        products.add(product)
        if gap <= H24:
            cnt_24h += 1
            amt_24h += old_amt
            if gap <= H1:
                cnt_1h += 1

    card_1h, card_24h = 0, 0
    card_uids = set()
    for old_t, old_amt, old_uid in reversed(c["recent"]):
        gap = t - old_t
        if gap > H24:
            break
        card_24h += 1
        card_uids.add(old_uid)
        if gap <= H1:
            card_1h += 1

    return {
        "uid_cnt_1h": cnt_1h,
        "uid_cnt_24h": cnt_24h,
        "uid_cnt_7d": cnt_7d,
        "uid_amt_24h": amt_24h,
        "uid_hist_cnt": u["n"],
        "uid_amt_mean": u["total"] / u["n"] if u["n"] else np.nan,
        "uid_amt_ratio": amt / (u["total"] / u["n"]) if u["n"] >= 2 and u["total"] > 0 else np.nan,
        "uid_secs_since_last": t - u["last_t"] if u["last_t"] is not None else np.nan,
        "uid_new_device": int(device is not None and u["n"] > 0 and device not in u["devices"]),
        "uid_new_email": int(email is not None and u["n"] > 0 and email not in u["emails"]),
        "uid_distinct_products_7d": len(products),
        "card_cnt_1h": card_1h,
        "card_cnt_24h": card_24h,
        "card_distinct_uid_24h": len(card_uids),
        "card_amt_ratio": amt / (c["total"] / c["n"]) if c["n"] >= 2 and c["total"] > 0 else np.nan,
        "card_new_addr": int(addr is not None and c["n"] > 0 and addr not in c["addrs"]),
    }


def update_history(p, uid, card_key):
    t = float(p["TransactionDT"])
    amt = float(p["TransactionAmt"])
    u = state["customers"][uid]
    c = state["cards"][card_key]
    u["recent"].append((t, amt, p["ProductCD"]))
    u["n"] += 1
    u["total"] += amt
    u["last_t"] = t
    if p["DeviceInfo"] is not None:
        u["devices"].add(p["DeviceInfo"])
    if p["P_emaildomain"] is not None:
        u["emails"].add(p["P_emaildomain"])
    c["recent"].append((t, amt, uid))
    c["n"] += 1
    c["total"] += amt
    if p["addr1"] is not None:
        c["addrs"].add(p["addr1"])


def model_row(p, window):
    row = dict(window)
    amt = p["TransactionAmt"]
    row["log_amt"] = np.log1p(amt)
    row["hour"] = (p["TransactionDT"] // 3600) % 24
    row["email_match"] = np.nan if p["R_emaildomain"] is None else float(p["P_emaildomain"] == p["R_emaildomain"])
    row["has_identity"] = int(p["DeviceType"] is not None)
    for col, mapping in state["category_maps"].items():
        row[col + "_code"] = mapping.get(p[col], np.nan)
    for col in state["features"]:
        if col not in row:
            row[col] = p[col]
    return pd.DataFrame([row])[state["features"]].astype(float)


def top_reasons(X):
    contrib = state["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    order = np.argsort(-contrib)[:3]
    return [state["features"][i] for i in order if contrib[i] > 0]


def load_history():
    cols = ["TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "D1", "DeviceInfo", "P_emaildomain"]
    hist = pd.read_parquet(HISTORY_PATH, columns=cols)
    hist = hist.astype(object).where(hist.notna(), None)
    n = 0
    for p in hist.to_dict("records"):
        uid, card_key = make_keys(p)
        update_history(p, uid, card_key)
        n += 1
    return n


@asynccontextmanager
async def lifespan(app):
    bundle = joblib.load(MODEL_PATH)
    state.update(bundle)
    state["customers"] = defaultdict(new_customer)
    state["cards"] = defaultdict(new_card)
    state["seen"] = set()
    t0 = time.time()
    state["history_rows"] = load_history()
    state["warm_seconds"] = round(time.time() - t0, 1)
    yield
    state.clear()


app = FastAPI(title="Fraud scoring API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": state["model_name"], "history_rows": state["history_rows"],
            "warm_seconds": state["warm_seconds"], "customers": len(state["customers"])}


@app.post("/score")
def score(payment: Payment):
    p = payment.model_dump()
    if p["TransactionID"] in state["seen"]:
        raise HTTPException(status_code=409, detail="payment already scored")
    state["seen"].add(p["TransactionID"])

    uid, card_key = make_keys(p)
    window = window_features(p, uid, card_key)
    X = model_row(p, window)
    risk = float(state["model"].predict_proba(X)[0, 1])
    update_history(p, uid, card_key)

    if risk >= state["thr_block"]:
        decision = "BLOCK"
    elif risk >= state["thr_review"]:
        decision = "STEP_UP"
    else:
        decision = "APPROVE"

    reasons = top_reasons(X) if decision != "APPROVE" else []
    return {"TransactionID": p["TransactionID"], "risk_score": round(risk, 6), "decision": decision, "reasons": reasons}
