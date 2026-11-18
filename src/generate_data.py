"""Synthetic UPI transaction generator. No real data, no real accounts."""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

START = datetime(2026, 10, 1)
DAYS = 30
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata",
          "Pune", "Jaipur", "Ahmedabad", "Lucknow", "Chandigarh", "Kochi"]
CHANNELS = ["UPI_APP", "QR_SCAN", "WEB", "COLLECT_REQUEST"]
TXN_TYPES = ["P2P", "P2M", "BILL_PAY"]
LEGIT_BENS = [f"B{i:04d}" for i in range(1, 601)]
MULE_BENS = [f"B{i:04d}" for i in range(601, 616)]   # hidden mule accounts


# ---------- helpers ----------
def rand_ts(day=None, hour=None):
    d = int(rng.integers(0, DAYS)) if day is None else int(day)
    h = int(rng.integers(0, 24)) if hour is None else int(hour)
    return START + timedelta(days=d, hours=h,
                             minutes=int(rng.integers(0, 60)),
                             seconds=int(rng.integers(0, 60)))


def new_ben(u):
    while True:
        b = str(rng.choice(LEGIT_BENS))
        if b not in u["known"]:
            return b


def mk(u, ts, amt, ben, fraud=0, scenario="NORMAL",
       device=None, city=None, channel=None, ttype=None):
    return {
        "user_id": u["user_id"],
        "beneficiary_id": ben,
        "amount": round(float(amt), 2),
        "timestamp": ts,
        "device_id": device or str(rng.choice(u["devices"])),
        "city": city or u["city"],
        "channel": channel or str(rng.choice(CHANNELS, p=[0.6, 0.25, 0.1, 0.05])),
        "txn_type": ttype or str(rng.choice(TXN_TYPES, p=[0.5, 0.35, 0.15])),
        "is_fraud": fraud,
        "scenario": scenario,
    }


def make_users(n):
    users = []
    for i in range(1, n + 1):
        users.append({
            "user_id": f"U{i:04d}",
            "city": str(rng.choice(CITIES)),
            "devices": [f"D{i:04d}-{k}" for k in range(int(rng.integers(1, 3)))],
            "mean_amt": float(rng.lognormal(6.2, 0.5)),     # typical amount ~ Rs 500
            "rate": float(rng.uniform(0.5, 3.0)),           # txns per day
            "h0": int(rng.integers(7, 10)),                 # active hours start
            "h1": int(rng.integers(20, 24)),                # active hours end
            "known": [str(b) for b in rng.choice(
                LEGIT_BENS, size=int(rng.integers(3, 9)), replace=False)],
        })
    return users


# ---------- normal behaviour ----------
def normal_txns(u):
    out = []
    for d in range(DAYS):
        for _ in range(int(rng.poisson(u["rate"]))):
            ts = rand_ts(d, int(rng.integers(u["h0"], u["h1"])))
            amt = max(10, rng.lognormal(np.log(u["mean_amt"]), 0.4))
            if rng.random() < 0.9:
                out.append(mk(u, ts, amt, str(rng.choice(u["known"]))))
            else:  # paying someone new, small amount: legit
                out.append(mk(u, ts, min(amt, u["mean_amt"]), new_ben(u),
                              0, "LEGIT_NEW_BEN"))
    return out


# ---------- fraud scenarios ----------
def f_velocity(u):
    t, out = rand_ts(), []
    for _ in range(int(rng.integers(6, 12))):
        t += timedelta(seconds=int(rng.integers(5, 40)))
        out.append(mk(u, t, u["mean_amt"] * rng.uniform(1, 3), new_ben(u),
                      1, "VELOCITY_BURST"))
    return out


def f_high_amount(u):
    ts = rand_ts(hour=int(rng.integers(u["h0"], u["h1"])))
    return [mk(u, ts, u["mean_amt"] * rng.uniform(10, 25), new_ben(u),
               1, "HIGH_AMOUNT")]


def f_new_ben_high(u):
    t, ben = rand_ts(), new_ben(u)
    test = mk(u, t, rng.uniform(1, 10), ben, 1, "NEW_BENEFICIARY_HIGH_VALUE")
    t2 = t + timedelta(minutes=int(rng.integers(10, 60)))
    big = mk(u, t2, u["mean_amt"] * rng.uniform(8, 15), ben,
             1, "NEW_BENEFICIARY_HIGH_VALUE")
    return [test, big]


def f_ato(u):
    t, out = rand_ts(), []
    dev = f"D-NEW-{int(rng.integers(10000, 99999))}"
    city = str(rng.choice([c for c in CITIES if c != u["city"]]))
    for _ in range(int(rng.integers(3, 7))):
        t += timedelta(minutes=int(rng.integers(1, 6)))
        out.append(mk(u, t, u["mean_amt"] * rng.uniform(5, 12), new_ben(u),
                      1, "ACCOUNT_TAKEOVER", device=dev, city=city,
                      channel="UPI_APP"))
    return out


def f_mule(users):
    mule, t0, out = str(rng.choice(MULE_BENS)), rand_ts(), []
    victims = rng.choice(len(users), size=int(rng.integers(8, 15)), replace=False)
    for idx in victims:
        t = t0 + timedelta(minutes=int(rng.integers(0, 180)))
        out.append(mk(users[int(idx)], t, rng.uniform(2000, 9000), mule,
                      1, "MULE_FANIN"))
    return out


def f_structuring(u):
    t, ben, out = rand_ts(), new_ben(u), []
    for _ in range(int(rng.integers(4, 9))):
        t += timedelta(minutes=int(rng.integers(5, 20)))
        out.append(mk(u, t, rng.uniform(9000, 9990), ben, 1, "STRUCTURING"))
    return out


def f_odd_hour(u):
    t, out = rand_ts(hour=int(rng.integers(2, 5))), []
    for _ in range(int(rng.integers(1, 4))):
        t += timedelta(minutes=int(rng.integers(2, 15)))
        out.append(mk(u, t, u["mean_amt"] * rng.uniform(3, 8), new_ben(u),
                      1, "ODD_HOUR"))
    return out


# ---------- legitimate control cases (hard negatives) ----------
def l_festival(u):
    ts = rand_ts(hour=int(rng.integers(11, 20)))
    return [mk(u, ts, u["mean_amt"] * rng.uniform(5, 10),
               str(rng.choice(u["known"])), 0, "LEGIT_FESTIVAL_SPIKE", ttype="P2M")]


def l_travel(u):
    t, out = rand_ts(hour=10), []
    city = str(rng.choice([c for c in CITIES if c != u["city"]]))
    for _ in range(int(rng.integers(3, 7))):
        t += timedelta(hours=int(rng.integers(1, 3)))
        out.append(mk(u, t, rng.lognormal(np.log(u["mean_amt"]), 0.3),
                      str(rng.choice(u["known"])), 0, "LEGIT_TRAVEL", city=city))
    return out


def l_new_device(u):
    t, out = rand_ts(hour=11), []
    dev = f"D-NEW-{int(rng.integers(10000, 99999))}"
    for _ in range(int(rng.integers(2, 5))):
        t += timedelta(hours=int(rng.integers(1, 4)))
        out.append(mk(u, t, rng.lognormal(np.log(u["mean_amt"]), 0.3),
                      str(rng.choice(u["known"])), 0, "LEGIT_NEW_DEVICE", device=dev))
    return out


# ---------- main ----------
def generate(n_users=300):
    users = make_users(n_users)
    pick = lambda: users[int(rng.integers(len(users)))]
    rows = []
    for u in users:
        rows += normal_txns(u)

    plan = [(f_velocity, 25), (f_high_amount, 30), (f_new_ben_high, 25),
            (f_ato, 20), (f_structuring, 15), (f_odd_hour, 20),
            (l_festival, 40), (l_travel, 15), (l_new_device, 15)]
    for fn, count in plan:
        for _ in range(count):
            rows += fn(pick())
    for _ in range(6):
        rows += f_mule(users)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i + 1:06d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = generate()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"Saved {len(df)} transactions to data/transactions.csv")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}\n")
    print(df["scenario"].value_counts())