from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
Villani / OT rad 03 v3 — displacement interpolacija → next

π* = Sinkhorn(p_prev, p_now), C_ij=|i−j|
McCann / displacement (diskretno):
  za t∈(0,1): masa π_ij ide na „međutačku“ k = round((1−t)·i + t·j)
  μ_t = marginal te mase na {1…39}

Predikcija: μ_{t=1.5} ekstrapolacija duž istog smera (p_now → napred)
  skor = μ_pred − p_glob (+ blagi p_now excess)
Ban last; next. CSV ceo, seed=39.
Ime: ig_Villani_03_v3_displacement.py
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
WINDOW = 100
EPS_OT = 0.05
SINKHORN_ITERS = 200
T_PRED = 1.5  # t=0 p_prev, t=1 p_now, t>1 ekstrapolacija
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def window_p(draws: np.ndarray, end: int, w: int = WINDOW) -> np.ndarray:
    start = max(0, end - w)
    chunk = draws[start:end]
    cnt = Counter(chunk.reshape(-1).tolist())
    n_slots = max(len(chunk) * FRONT_SELECT, 1)
    p = np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    p = np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)
    return p / p.sum()


def cost_abs() -> np.ndarray:
    idx = np.arange(1, FRONT_N + 1, dtype=float)
    return np.abs(idx[:, None] - idx[None, :])


def sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float = EPS_OT,
    n_iter: int = SINKHORN_ITERS,
) -> np.ndarray:
    K = np.exp(-C / eps)
    u = np.ones(FRONT_N)
    v = np.ones(FRONT_N)
    for _ in range(n_iter):
        u = a / np.clip(K @ v, 1e-18, None)
        v = b / np.clip(K.T @ u, 1e-18, None)
    return (u[:, None] * K) * v[None, :]


def displacement_measure(pi: np.ndarray, t: float) -> np.ndarray:
    """
    Diskretni McCann: masa π[i,j] → indeks round((1−t)·i + t·j), clamp 0..38.
    """
    mu = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        for j in range(FRONT_N):
            m = pi[i, j]
            if m <= 0:
                continue
            pos = (1.0 - t) * i + t * j
            k = int(round(pos))
            k = max(0, min(FRONT_N - 1, k))
            mu[k] += m
    s = mu.sum()
    if s <= 0:
        return np.ones(FRONT_N) / FRONT_N
    return mu / s


def number_scores(
    mu_pred: np.ndarray, p_now: np.ndarray, p_glob: np.ndarray, ban: set[int]
) -> dict[int, float]:
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = float(
                (mu_pred[i] - p_glob[i]) + 0.10 * (p_now[i] - p_glob[i])
            )
    return out


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban):
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws, score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_03_v3(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    n = len(draws)
    p_prev = window_p(draws, n - 1, WINDOW)
    p_now = window_p(draws, n, WINDOW)
    p_glob = global_p(draws)
    C = cost_abs()
    pi = sinkhorn(p_prev, p_now, C)
    mu_pred = displacement_measure(pi, T_PRED)
    # dijagnostika: μ na t=0/1 treba ≈ p_prev / p_now
    mu0 = displacement_measure(pi, 0.0)
    mu1 = displacement_measure(pi, 1.0)
    score = number_scores(mu_pred, p_now, p_glob, ban)
    combo = predict_next(draws, score, ban)

    print(f"CSV: {csv_path.name}")
    print(
        f"Kola: {n} | seed={SEED} | W={WINDOW} t={T_PRED} | ig_Villani_03_v3 displace"
    )
    print(f"last: {last.tolist()}")
    print()
    print("=== displacement ===")
    print(
        {
            "L1_mu0_vs_pprev": round(float(np.abs(mu0 - p_prev).sum()), 6),
            "L1_mu1_vs_pnow": round(float(np.abs(mu1 - p_now).sum()), 6),
            "L1_mupred_vs_pnow": round(float(np.abs(mu_pred - p_now).sum()), 6),
        }
    )
    print()
    ranked = sorted(
        ((n_, float(score[n_])) for n_ in range(1, FRONT_N + 1) if n_ not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (μ_pred − p_glob) ===")
    print([(n_, round(sc, 6)) for n_, sc in ranked[:12]])
    print()
    print("=== next (ig_Villani_03_v3 displace) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_ig_03_v3()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | W=100 t=1.5 | ig_Villani_03_v3 displace
last: [4, 5, 6, 11, 12, 18, 28]

=== displacement ===
{'L1_mu0_vs_pprev': 0.011569, 'L1_mu1_vs_pnow': 0.0, 'L1_mupred_vs_pnow': 0.004264}

=== top12 skor (μ_pred − p_glob) ===
[(1, 0.015562), (29, 0.010427), (14, 0.008702), (27, 0.007806), (16, 0.007722), (24, 0.007654), (8, 0.006793), (38, 0.004444), (20, 0.0037), (31, 0.003247), (19, 0.002248), (34, 0.001822)]

=== next (ig_Villani_03_v3 displace) ===
next: [3, x, 19, y, 24, z, 34]
"""



"""
McCann duž π*: μ na t=1.5 → skor → next.
"""



"""
Rad	    Tema
03      Villani / Optimal Transport
04      Monge controller
05      Lie groups + observability





#	Jezgro	Fajlovi

01 - 6
Stošić / Fisher
ig_Stosic_01_fisher_v1_empfreq … v6_banlast

02 - 23
Stošić / IG (chart–Γ–ekscitacija–χ–τ–…)
ig_Stosic_02_v1_multichart … v23_permnull

03 - 3
Villani / OT
ig_Villani_03_v1_sinkhorn, v2_w1path, v3_displacement

04 - 2
Monge kontroler
ig_Monge_04_v1_map, v2_iterctrl

05 - 3
Lie + observability
ig_Lie_05_v1_genobs, v2_bracket, v3_gramrank
"""
