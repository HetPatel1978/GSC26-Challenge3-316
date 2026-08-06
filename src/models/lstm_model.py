"""
LSTM sequence model for imminent-failure prediction: instead of a single
30-min window's summary stats (the tabular models' input), each example is
the last SEQ_LEN windows (default 6 = 3 hours) of a task's usage history,
predicting whether the task fails in the 30 minutes after the most recent
window -- same label definition as the other models.

Scale note: building fixed-length lag sequences for all 66.6M window-rows
via a per-task shift is a >20GB intermediate (SEQ_LEN x 17 features x 66.6M
rows), too large for this machine. Instead we bound the task universe: every
task that ever has a positive window, plus a random sample of
NEG_TASK_RATIO x as many negative-only tasks, then build sequences from
*all* windows of just those tasks. This means the LSTM's test set is a
(large, but not 100%) subsample of the tasks the tabular models are scored
on -- noted here and in docs/03-BASELINE-RESULTS.md rather than silently
treated as identical.

Tasks with fewer than SEQ_LEN preceding windows are zero-padded at the
start of the sequence (a simplification -- no explicit padding mask is fed
to the LSTM, so very short histories look like "quiet" history rather than
"unknown" history).

Usage:
    python src/models/lstm_model.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.dataset import FEATURE_COLS, FEATURES_PATH, NULLABLE_COLS  # noqa: E402
from eval.metrics import summarize, save_result, print_metrics  # noqa: E402

SEQ_LEN = 6  # 6 x 30min = 3h lookback
NEG_TASK_RATIO = 10
TRAIN_NEG_WINDOW_RATIO = 20.0  # same undersampling ratio as the tabular baselines
MODEL_PATH = REPO_ROOT / "models" / "lstm.pt"
SEED = 42


def select_task_subset(df: pl.DataFrame) -> pl.DataFrame:
    task_cols = ["job_id", "task_index"]
    pos_tasks = df.filter(pl.col("label_fail_soon") == 1).select(task_cols).unique()
    all_tasks = df.select(task_cols).unique()
    neg_pool = all_tasks.join(pos_tasks, on=task_cols, how="anti")
    n_neg = min(neg_pool.height, pos_tasks.height * NEG_TASK_RATIO)
    neg_tasks = neg_pool.sample(n=n_neg, seed=SEED)
    keep = pl.concat([pos_tasks, neg_tasks])
    print(f"task subset: {pos_tasks.height:,} positive-containing + {n_neg:,} negative-only "
          f"= {keep.height:,} / {all_tasks.height:,} total tasks")
    return df.join(keep, on=task_cols, how="inner")


def build_sequences(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["job_id", "task_index", "window_start"])
    exprs = []
    for i in range(SEQ_LEN):
        lag = SEQ_LEN - 1 - i  # i=0 -> earliest step, i=SEQ_LEN-1 -> current/anchor window (lag=0)
        for c in FEATURE_COLS:
            exprs.append(pl.col(c).shift(lag).over(["job_id", "task_index"]).alias(f"{c}__t{i}"))
    return df.with_columns(exprs)


def to_tensor(df: pl.DataFrame) -> np.ndarray:
    """(n_rows, SEQ_LEN, n_features), zero-padded where lag history is missing."""
    n = df.height
    X = np.zeros((n, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32)
    for i in range(SEQ_LEN):
        cols = [f"{c}__t{i}" for c in FEATURE_COLS]
        X[:, i, :] = df.select(cols).fill_null(0.0).to_numpy()
    return X


def predict_proba_batched(model: nn.Module, X: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(X[start:start + batch_size])
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading features ...")
    df = pl.read_parquet(FEATURES_PATH)
    medians = {c: df[c].median() for c in NULLABLE_COLS}
    df = df.with_columns([pl.col(c).fill_null(v) for c, v in medians.items()])

    cutoff = df["window_end"].quantile(0.8)

    df = select_task_subset(df)
    print("building lag sequences ...")
    seq_df = build_sequences(df)

    train_df = seq_df.filter(pl.col("window_end") <= cutoff)
    test_df = seq_df.filter(pl.col("window_end") > cutoff)

    pos = train_df.filter(pl.col("label_fail_soon") == 1)
    neg = train_df.filter(pl.col("label_fail_soon") == 0)
    n_neg_keep = min(neg.height, int(pos.height * TRAIN_NEG_WINDOW_RATIO))
    neg_sampled = neg.sample(n=n_neg_keep, seed=SEED)
    train_bal = pl.concat([pos, neg_sampled]).sample(fraction=1.0, seed=SEED, shuffle=True)

    print(f"train sequences: {train_bal.height:,} ({pos.height:,} pos / {n_neg_keep:,} neg) | "
          f"test sequences: {test_df.height:,} ({int(test_df['label_fail_soon'].sum()):,} pos)")

    X_train = to_tensor(train_bal)
    y_train = train_bal["label_fail_soon"].to_numpy().astype(np.float32)
    X_test = to_tensor(test_df)
    y_test = test_df["label_fail_soon"].to_numpy().astype(np.float32)

    # standardize using train statistics (flatten over time steps)
    flat = X_train.reshape(-1, X_train.shape[-1])
    mu, sigma = flat.mean(axis=0), flat.std(axis=0)
    sigma[sigma == 0] = 1e-6
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X_train))
    n_val = int(0.1 * len(idx))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    Xt_fit = torch.tensor(X_train[fit_idx], device=device)
    yt_fit = torch.tensor(y_train[fit_idx], device=device)
    Xt_val = torch.tensor(X_train[val_idx], device=device)
    yt_val = torch.tensor(y_train[val_idx], device=device)
    Xt_test = torch.tensor(X_test, device=device)

    model = LSTMClassifier(len(FEATURE_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    batch_size = 4096
    n_fit = len(Xt_fit)
    best_val_loss, best_state, patience, bad_epochs = float("inf"), None, 5, 0

    for epoch in range(50):
        model.train()
        perm = torch.randperm(n_fit, device=device)
        total_loss = 0.0
        for start in range(0, n_fit, batch_size):
            batch = perm[start:start + batch_size]
            xb, yb = Xt_fit[batch], yt_fit[batch]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / n_fit

        model.eval()
        with torch.no_grad():
            val_logits = model(Xt_val)
            val_loss = loss_fn(val_logits, yt_val).item()
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    train_proba = predict_proba_batched(model, Xt_fit)
    best_t, best_f1 = None, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y_train[fit_idx], (train_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"tuned threshold (max train F1): {best_t:.2f} (train F1={best_f1:.4f})")

    test_proba = predict_proba_batched(model, Xt_test)
    y_pred = (test_proba >= best_t).astype(int)

    metrics = summarize(y_test.astype(int), y_pred, test_proba)
    print_metrics("lstm", metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mu": mu, "sigma": sigma, "seq_len": SEQ_LEN, "feature_cols": FEATURE_COLS,
    }, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    save_result("lstm", metrics, extra={
        "threshold": float(best_t), "seq_len": SEQ_LEN,
        "n_train_tasks_subsample_ratio": NEG_TASK_RATIO,
        "note": "trained/evaluated on a task subsample (all positive-containing tasks + "
                f"{NEG_TASK_RATIO}x random negative-only tasks), not the full task universe "
                "the tabular models use -- see docs/03-BASELINE-RESULTS.md",
        "cutoff_window_end_us": int(cutoff),
    })


if __name__ == "__main__":
    main()
