"""
sweep_ringattractor.py
======================
Two-phase parameter sweep for the ring attractor simulation.

PHASE 1: Coarse random search across the full parameter space.
PHASE 2: Fine grid search zoomed into the best region found in Phase 1.

Success metric: agent commits to one target cleanly.
  - straightness_score: ratio of beeline distance to actual path length (1.0 = perfectly straight)
  - time_to_target:     timestep at which agent first gets within `arrival_radius` of any target
  - composite_score:    weighted combination of the two (higher = better)

Usage
-----
  python sweep_ringattractor.py

Outputs
-------
  sweep_results.csv   — full results from both phases
  sweep_summary.txt   — top 10 parameter combos with scores
"""

import numpy as np
import pandas as pd
import itertools
import time
import simulate_ringattractor
from multiprocessing import Pool, cpu_count

# ─────────────────────────────────────────────
# SIMULATION GEOMETRY (keep fixed during sweep)
# ─────────────────────────────────────────────

N         = 100
L         = 100
T         = 5000
nagents   = 1
ntargets  = 2
dt        = 0.1
v0        = 0.05
rEgo      = 0
rEgoTarget= 0
Egonumber = 1
periodic_flag = 0
distf     = 1
adistf    = 1
hColl     = -10
rColl     = 0

initialx  = np.array([L / 2])
initialy  = np.array([L / 2])
initialxt = np.array([L - 30, L - 30])
initialyt = np.array([L / 2 + 15, L / 2 - 15])
v0t       = np.zeros(ntargets)

# precompute J (expensive, do once)
nu    = 0.5
theta = np.linspace(0, 2 * np.pi, N + 1)[:-1]
J     = np.zeros((N, N))
for i in range(N):
    deltah   = np.abs(theta - theta[i])
    deltah   = np.pi - np.abs(np.pi - deltah)
    J[i, :]  = np.cos(np.pi * (deltah / np.pi) ** nu)
    J[i, i]  = 0.0
J = np.squeeze(J)

# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

ARRIVAL_RADIUS   = 8.0   # how close the agent needs to get to "reach" a target
STRAIGHTNESS_W   = 0.9   # weight for straightness in composite score
TIME_W           = 0.1   # weight for speed in composite score

# idk if we care about time, maybe just if the trajectory is about the same?


def score_run(xPos, yPos, targetsx, targetsy):
    """
    Returns a dict of metrics for one simulation run.

    straightness_score : float in [0, 1]
        Beeline distance / actual path length.
        1.0 means the agent went perfectly straight.

    time_to_target : int or None
        Timestep when agent first enters ARRIVAL_RADIUS of any target.
        None means the agent never reached a target.

    reached_target : bool

    composite_score : float in [0, 1]  (higher = better)
        Weighted combination. Returns 0 if agent never reached a target.
    """
    xs = xPos[0, :]
    ys = yPos[0, :]

    # path length
    dx       = np.diff(xs)
    dy       = np.diff(ys)
    steps    = np.sqrt(dx**2 + dy**2)
    path_len = np.sum(steps)

    # straightness: beeline from start to final position / path length
    # we actually don't want it on a beeline - we'd rather have it at a bit of an angle
    beeline = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    if path_len < 1e-9:
        straightness = 0.0
    else:
        straightness = min(beeline / path_len, 1.0)

    # time to reach any target
    time_reached = None
    for ttarg in range(ntargets):
        tx = targetsx[ttarg, :]
        ty = targetsy[ttarg, :]
        dists = np.sqrt((xs - tx)**2 + (ys - ty)**2)
        hits  = np.where(dists < ARRIVAL_RADIUS)[0]
        if len(hits) > 0:
            t = int(hits[0])
            if time_reached is None or t < time_reached:
                time_reached = t

    reached = time_reached is not None

    if not reached:
        composite = 0.0
    else:
        # normalise time: 0 steps = 1.0, T steps = 0.0
        time_score = 1.0 - (time_reached / T)
        composite  = STRAIGHTNESS_W * straightness + TIME_W * time_score

    return {
        "straightness": round(straightness, 4),
        "time_to_target": time_reached,
        "reached_target": reached,
        "composite_score": round(composite, 4),
    }


# ─────────────────────────────────────────────
# SINGLE RUN WRAPPER
# ─────────────────────────────────────────────

def run_one(params):
    """
    params: dict with keys h0_val, h_b_val, sigma_val, allocentric_val, seed
    Returns params dict extended with score metrics.
    """
    h0_val       = params["h0_val"]
    h_b_val      = params["h_b_val"]
    sigma_val    = params["sigma_val"]
    alloc        = params["allocentric_val"]
    seed         = params["seed"]

    np.random.seed(seed)

    h0 = np.full(ntargets + nagents, h0_val)

    try:
        headings, xPos, yPos, targetsx, targetsy = simulate_ringattractor.simulate_ring_attractor(
            N, L, T, ntargets, nagents, alloc, periodic_flag,
            rEgo, rEgoTarget, Egonumber,
            distf, adistf, J, beta=100,
            h0=h0, h_b=h_b_val, dt=dt, v0=v0, v0t=v0t,
            sigma=sigma_val, hColl=hColl, rColl=rColl,
            initialx=initialx, initialy=initialy,
            initialxt=initialxt, initialyt=initialyt,
            plot=False,
        )
        metrics = score_run(xPos, yPos, targetsx, targetsy)
    except Exception as e:
        metrics = {
            "straightness": 0.0,
            "time_to_target": None,
            "reached_target": False,
            "composite_score": 0.0,
            "error": str(e),
        }

    return {**params, **metrics}


# ─────────────────────────────────────────────
# PHASE 1 — COARSE RANDOM SEARCH
# ─────────────────────────────────────────────

PHASE1_N_SAMPLES  = 120   # number of random combos to try
PHASE1_SEEDS      = 3     # runs per combo (averaged) to reduce noise
PHASE1_TOP_K      = 10    # how many combos to carry into Phase 2

PHASE1_RANGES = {
    "h0_val":          (0.2,  0.7),
    "h_b_val":         (0.0, 0.4),
    "sigma_val":       (0, 0.8),
    "allocentric_val": [0, 1],       # discrete
}


def phase1_sample():
    """Draw one random parameter combo from Phase 1 ranges."""
    return {
        "h0_val":          round(np.random.uniform(*PHASE1_RANGES["h0_val"]), 4),
        "h_b_val":         round(np.random.uniform(*PHASE1_RANGES["h_b_val"]), 4),
        "sigma_val":       round(np.random.uniform(*PHASE1_RANGES["sigma_val"]), 4),
        "allocentric_val": int(np.random.choice(PHASE1_RANGES["allocentric_val"])),
    }


def run_phase1(n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"\n{'='*55}")
    print(f"PHASE 1: coarse random search ({PHASE1_N_SAMPLES} combos × {PHASE1_SEEDS} seeds)")
    print(f"Workers: {n_workers}")
    print(f"{'='*55}")

    # build job list
    jobs = []
    for _ in range(PHASE1_N_SAMPLES):
        combo = phase1_sample()
        for s in range(PHASE1_SEEDS):
            jobs.append({**combo, "seed": s, "phase": 1})

    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(run_one, jobs)
    elapsed = time.time() - t0
    print(f"Phase 1 done in {elapsed:.1f}s")

    df = pd.DataFrame(results)

    # average over seeds for each combo
    group_cols = ["h0_val", "h_b_val", "sigma_val", "allocentric_val"]
    agg = (
        df.groupby(group_cols)
        .agg(
            composite_score=("composite_score", "mean"),
            straightness=("straightness", "mean"),
            reached_pct=("reached_target", "mean"),
            avg_time=("time_to_target", "mean"),
        )
        .reset_index()
        .sort_values("composite_score", ascending=False)
    )

    return df, agg


# ─────────────────────────────────────────────
# PHASE 2 — FINE GRID SEARCH
# ─────────────────────────────────────────────

PHASE2_GRID_POINTS = 4    # values per continuous dimension
PHASE2_SEEDS       = 5    # more seeds for reliable estimates
PHASE2_WINDOW      = 0.15  # ± half-width around best Phase 1 values


def phase2_grid(top_combos):
    """
    Build a fine grid centered on the best Phase 1 results.
    For each continuous param, take the mean of top_k values ± PHASE2_WINDOW.
    allocentric_val stays discrete; we try both if both appeared in top_k.
    """
    best_h0    = top_combos["h0_val"].mean()
    best_hb    = top_combos["h_b_val"].mean()
    best_sigma = top_combos["sigma_val"].mean()
    alloc_vals = top_combos["allocentric_val"].unique().tolist()

    def fine_range(center, lo_global, hi_global):
        lo = max(lo_global, center - PHASE2_WINDOW)
        hi = min(hi_global, center + PHASE2_WINDOW)
        return np.round(np.linspace(lo, hi, PHASE2_GRID_POINTS), 4).tolist()

    h0_vals    = fine_range(best_h0,    *PHASE1_RANGES["h0_val"])
    hb_vals    = fine_range(best_hb,    *PHASE1_RANGES["h_b_val"])
    sigma_vals = fine_range(best_sigma, *PHASE1_RANGES["sigma_val"])

    jobs = []
    for h0, hb, sig, alloc, seed in itertools.product(
        h0_vals, hb_vals, sigma_vals, alloc_vals, range(PHASE2_SEEDS)
    ):
        jobs.append({
            "h0_val": h0, "h_b_val": hb, "sigma_val": sig,
            "allocentric_val": alloc, "seed": seed, "phase": 2
        })
    return jobs


def run_phase2(top_combos, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    jobs = phase2_grid(top_combos)
    n_combos = len(jobs) // PHASE2_SEEDS

    print(f"\n{'='*55}")
    print(f"PHASE 2: fine grid search ({n_combos} combos × {PHASE2_SEEDS} seeds)")
    print(f"Workers: {n_workers}")
    print(f"{'='*55}")

    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(run_one, jobs)
    elapsed = time.time() - t0
    print(f"Phase 2 done in {elapsed:.1f}s")

    df = pd.DataFrame(results)
    group_cols = ["h0_val", "h_b_val", "sigma_val", "allocentric_val"]
    agg = (
        df.groupby(group_cols)
        .agg(
            composite_score=("composite_score", "mean"),
            straightness=("straightness", "mean"),
            reached_pct=("reached_target", "mean"),
            avg_time=("time_to_target", "mean"),
        )
        .reset_index()
        .sort_values("composite_score", ascending=False)
    )
    return df, agg


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Phase 1 ──
    df1, agg1 = run_phase1()
    top_k = agg1.head(PHASE1_TOP_K)

    print("\nTop combos from Phase 1:")
    print(top_k.to_string(index=False))

    # ── Phase 2 ──
    df2, agg2 = run_phase2(top_k)

    print("\nTop combos from Phase 2:")
    print(agg2.head(10).to_string(index=False))

    # ── Save results ──
    all_raw = pd.concat([df1, df2], ignore_index=True)
    all_raw.to_csv("sweep_results.csv", index=False)

    with open("sweep_summary.txt", "w") as f:
        f.write("RING ATTRACTOR PARAMETER SWEEP — SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        f.write("Scoring weights:\n")
        f.write(f"  straightness : {STRAIGHTNESS_W}\n")
        f.write(f"  time score   : {TIME_W}\n")
        f.write(f"  arrival_radius: {ARRIVAL_RADIUS}\n\n")
        f.write("TOP 10 PARAMETER COMBOS (Phase 2):\n")
        f.write(agg2.head(10).to_string(index=False))
        f.write("\n\n")
        f.write("BEST SINGLE COMBO:\n")
        best = agg2.iloc[0]
        f.write(f"  h0_val           = {best.h0_val}\n")
        f.write(f"  h_b_val          = {best.h_b_val}\n")
        f.write(f"  sigma_val        = {best.sigma_val}\n")
        f.write(f"  allocentric_val  = {int(best.allocentric_val)}\n")
        f.write(f"  composite_score  = {best.composite_score:.4f}\n")
        f.write(f"  straightness     = {best.straightness:.4f}\n")
        f.write(f"  reached_pct      = {best.reached_pct:.2%}\n")
        f.write(f"  avg_time         = {best.avg_time:.1f}\n")

    print("\nSaved: sweep_results.csv, sweep_summary.txt")
    print(f"\nBEST PARAMETERS FOUND:")
    print(f"  h0_val           = {agg2.iloc[0].h0_val}")
    print(f"  h_b_val          = {agg2.iloc[0].h_b_val}")
    print(f"  sigma_val        = {agg2.iloc[0].sigma_val}")
    print(f"  allocentric_val  = {int(agg2.iloc[0].allocentric_val)}")
    print(f"  composite_score  = {agg2.iloc[0].composite_score:.4f}")
