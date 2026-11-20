"""Four-fold XGBoost workflow for Store Sales forecasting.

Select structural decisions and feature groups on B/C/D, tune 20 candidates on
B/C, select rounds on B/C/D, then evaluate A once and refit three test seeds.
All results are checkpointed so an interrupted run can be resumed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
from threading import Lock
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.build_features import BASE_FEATURES, PROC_DIR, ROOT, TARGET, apply_zero_series, load_fe
from src.features.guided_features import FEATURE_GROUPS, restrict_promo_leads
from src.models.xgboost import _normalize_numeric_categories

FOLDS = {"A": "2017-07-31", "B": "2017-07-15", "C": "2017-06-29", "D": "2016-07-31"}
DEV = ("B", "C", "D")
OIL = {"oil", "oil_ma7", "oil_ma14"}
LOCATION = {"city", "state", "store_type", "cluster"}
DEFAULT_PARAMS = {"objective": "reg:squarederror", "eval_metric": "rmse",
    "tree_method": "hist", "grow_policy": "lossguide", "max_depth": 0,
    "max_leaves": 127, "max_bin": 256, "eta": 0.05, "min_child_weight": 50,
    "reg_lambda": 5, "subsample": 0.8, "colsample_bytree": 0.7,
    "seed": 0, "nthread": 4}
STATE_LOCK = Lock()


def margin(frame):
    return (frame["rolling_mean_28"].fillna(frame["rolling_mean_7"])
            .fillna(frame["sales_lag_16"]).fillna(0).to_numpy(dtype="float32"))


def feature_columns(config):
    cols = [c for c in BASE_FEATURES if not (config.get("no_oil") and c in OIL)
            and not (config.get("drop_location") and c in LOCATION)]
    for group in config.get("groups", []):
        cols.extend(FEATURE_GROUPS[group])
    return cols


def inputs(frame, cols, config):
    data = frame[cols].copy()
    if config.get("numeric_store"):
        data["store_nbr"] = data["store_nbr"].astype("int16")
    return data


def qmatrix(frame, cols, config, *, label=False, ref=None, cutoff=None):
    kwargs = {"data": inputs(frame, cols, config), "enable_categorical": True,
              "max_bin": 256, "ref": ref}
    if label:
        kwargs["label"] = frame[TARGET].to_numpy(dtype="float32")
    if config.get("margin"):
        kwargs["base_margin"] = margin(frame)
    if config.get("decay") and cutoff is not None:
        age = (pd.Timestamp(cutoff) - frame["date"]).dt.days.to_numpy()
        kwargs["weight"] = np.power(0.5, age / 365).astype("float32")
    return xgb.QuantileDMatrix(**kwargs)


def train_fold(train, fold, config, params, rounds, patience, device):
    start = pd.Timestamp(FOLDS[fold]); end = start + pd.Timedelta(days=16)
    mask = train["date"].lt(start) & train[TARGET].notna()
    if config.get("start"):
        mask &= train["date"].ge(pd.Timestamp(config["start"]))
    tr = train.loc[mask]
    val = restrict_promo_leads(train.loc[train["date"].ge(start) & train["date"].lt(end)])
    val = val.loc[val[TARGET].notna()]
    if tr.empty or val.empty or val["date"].nunique() != 16:
        raise ValueError(f"Fold {fold} has incomplete train/validation dates")
    cols = feature_columns(config)
    dtr = qmatrix(tr, cols, config, label=True, cutoff=start)
    dva = qmatrix(val, cols, config, label=True, ref=dtr)
    actual_params = {**DEFAULT_PARAMS, **params, "device": device}
    bst = xgb.train(actual_params, dtr, num_boost_round=rounds, evals=[(dva, "val")],
                    early_stopping_rounds=patience, verbose_eval=False)
    if device == "cuda" and '"device":"cuda' not in bst.save_config():
        raise RuntimeError("XGBoost fell back to CPU; select --device cpu explicitly")
    pred = np.clip(bst.predict(dva, iteration_range=(0, bst.best_iteration + 1)), 0, None)
    score = float(np.sqrt(np.mean(np.square(pred - val[TARGET].to_numpy()))))
    return {"rmsle": score, "rounds": int(bst.best_iteration + 1), "n_train": len(tr),
            "n_val": len(val)}


def run_experiment(train, key, config, params, folds, state, state_path, rounds, patience, device):
    with STATE_LOCK:
        if key not in state["experiments"]:
            state["experiments"][key] = {"config": config, "params": params, "folds": {}}
        entry = state["experiments"][key]
        if entry["config"] != config or entry["params"] != params:
            raise ValueError(f"Checkpoint config changed for {key}")
    for fold in folds:
        if fold in entry["folds"]:
            continue
        print(f"{key} fold {fold}: training", flush=True)
        start = time.monotonic()
        result = train_fold(train, fold, config, params, rounds, patience, device)
        result["seconds"] = round(time.monotonic() - start, 1)
        with STATE_LOCK:
            entry["folds"][fold] = result
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{key} fold {fold}: {result['rmsle']:.6f}, {result['rounds']} rounds, "
              f"{result['seconds']}s", flush=True)
        gc.collect()
    scores = [entry["folds"][f]["rmsle"] for f in folds]
    return float(np.mean(scores))


def export(train, test, config, params, rounds, device, out_dir, sample_path):
    out_dir = Path(out_dir)
    cols = feature_columns(config)
    start = pd.Timestamp(FOLDS["A"])
    tr_mask = train["date"].lt(start) & train[TARGET].notna()
    if config.get("start"):
        tr_mask &= train["date"].ge(pd.Timestamp(config["start"]))
    va = restrict_promo_leads(train.loc[train["is_val"].eq(1)])
    te = restrict_promo_leads(test)
    actual = {**DEFAULT_PARAMS, **params, "device": device}
    print(f"Export validation: {rounds} rounds", flush=True)
    dtr = qmatrix(train.loc[tr_mask], cols, config, label=True, cutoff=start)
    dva = qmatrix(va, cols, config, ref=dtr)
    bst = xgb.train({**actual, "seed": 0}, dtr, num_boost_round=rounds, verbose_eval=False)
    vp = np.clip(bst.predict(dva), 0, None).astype("float32")
    scored = va[TARGET].notna().to_numpy()
    val_score = float(np.sqrt(np.mean((vp[scored] - va[TARGET].to_numpy()[scored]) ** 2)))
    del bst, dtr, dva
    gc.collect()

    train_mask = train[TARGET].notna()
    if config.get("start"):
        train_mask &= train["date"].ge(pd.Timestamp(config["start"]))
    cutoff = train["date"].max() + pd.Timedelta(days=1)
    dtr = qmatrix(train.loc[train_mask], cols, config, label=True, cutoff=cutoff)
    dte = qmatrix(te, cols, config, ref=dtr)
    preds = []
    model_dir = out_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1, 2):
        print(f"Refit seed {seed}: {rounds} rounds", flush=True)
        bst = xgb.train({**actual, "seed": seed}, dtr, num_boost_round=rounds, verbose_eval=False)
        preds.append(bst.predict(dte))
        bst.save_model(str(model_dir / f"xgb_guided_seed{seed}.ubj"))
    tp = apply_zero_series(np.clip(np.mean(preds, axis=0), 0, None), test).astype("float32")
    if len(vp) != 27664 or len(tp) != 28512 or not np.isfinite(vp).all() or not np.isfinite(tp).all():
        raise ValueError("Prediction shape or values invalid")
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "val" / "xgb_val.npy", vp)
    np.save(out_dir / "test" / "xgb_test.npy", tp)
    sample = pd.read_csv(sample_path)
    if not test["id"].is_unique or not sample["id"].is_unique or set(test["id"]) != set(sample["id"]):
        raise ValueError("Test ids do not match submission")
    by_id = pd.Series(np.expm1(tp.astype("float64")), index=test["id"].to_numpy())
    submission = pd.DataFrame({"id": sample["id"], "sales": by_id.reindex(sample["id"]).to_numpy()})
    sub_path = ROOT / "submissions" / "submission_xgb_guided.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Final fold A RMSLE: {val_score:.6f}; submission: {sub_path}", flush=True)
    return {"fold_a_rmsle": val_score, "rounds": rounds, "submission": str(sub_path)}


def run(proc_dir, out_dir, device, rounds, patience, trials, workers=1):
    out_dir = Path(out_dir); state_path = out_dir / "experiments" / "guided_results.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"experiments": {}}
    train, test = load_fe(proc_dir)
    _normalize_numeric_categories(train, test)
    base = {"margin": False, "groups": []}
    call = lambda name, cfg, params, folds: run_experiment(
        train, name, cfg, params, folds, state, state_path, rounds, patience, device)

    baseline = call("baseline", base, {}, DEV)
    base_margin = {**base, "margin": True}
    margin_score = call("base_margin", base_margin, {}, DEV)
    chosen = base_margin if margin_score < baseline else base
    print(f"Base margin selected: {chosen['margin']}", flush=True)

    candidates = {"selected_base": chosen,
        "no_oil": {**chosen, "no_oil": True},
        "from_2014": {**chosen, "start": "2014-01-01"},
        "from_2015": {**chosen, "start": "2015-01-01"},
        "decay": {**chosen, "decay": True},
        "numeric_store": {**chosen, "numeric_store": True},
        "drop_location": {**chosen, "drop_location": True}}
    structural = {name: call(f"structure_{name}", cfg, {}, DEV) for name, cfg in candidates.items()}
    best_name = min(structural, key=structural.get)
    chosen = candidates[best_name]
    best_score = structural[best_name]
    print(f"Structure selected: {best_name}, mean={best_score:.6f}", flush=True)
    for group in FEATURE_GROUPS:
        candidate = {**chosen, "groups": [*chosen["groups"], group]}
        score = call(f"feature_{group}", candidate, {}, DEV)
        if score < best_score:
            chosen, best_score = candidate, score
            print(f"Accepted feature group: {group}, mean={score:.6f}", flush=True)
        else:
            print(f"Rejected feature group: {group}, mean={score:.6f}", flush=True)

    rng = np.random.default_rng(42)
    trial_params = []
    for i in range(trials):
        params = {"eta": 0.08, "max_leaves": int(rng.choice([63, 127, 255])),
                  "min_child_weight": int(rng.integers(20, 101)),
                  "colsample_bytree": round(float(rng.uniform(0.5, 0.8)), 3),
                  "reg_lambda": round(float(rng.uniform(1, 20)), 3)}
        trial_params.append((i, params))
    candidates = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, f"tune_{i:02d}", chosen, params, ("B", "C")): (i, params)
                   for i, params in trial_params}
        for future in as_completed(futures):
            i, params = futures[future]
            candidates.append((future.result(), params, i))
    _, best_params, tune_id = min(candidates, key=lambda x: x[0])
    print(f"Tune selected: {tune_id}, params={best_params}", flush=True)
    final_params = {**best_params, "eta": 0.025}
    final_mean = call("final_dev", chosen, final_params, DEV)
    final_rounds = int(np.ceil(np.median([state["experiments"]["final_dev"]["folds"][f]["rounds"]
                                         for f in DEV]) * 1.05))
    print(f"Final development mean={final_mean:.6f}, refit rounds={final_rounds}", flush=True)
    result = export(train, test, chosen, final_params, final_rounds, device, out_dir,
                    ROOT / "data" / "raw" / "sample_submission.csv")
    state["selection"] = {"config": chosen, "tune_id": tune_id, "params": final_params,
                          "dev_mean": final_mean, **result}
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR / "xgb_guided")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel tuning trials; use 1 if memory is limited")
    args = parser.parse_args()
    if args.rounds < 1 or args.patience < 1 or args.trials < 1 or args.workers < 1:
        parser.error("rounds, patience, trials and workers must be positive")
    run(args.proc_dir, args.out_dir, args.device, args.rounds, args.patience, args.trials, args.workers)
