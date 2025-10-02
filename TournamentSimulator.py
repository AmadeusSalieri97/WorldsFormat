# sweep_formats_vs_headtohead_3formats.py
# ------------------------------------------------------------
# Compare three formats by Monte Carlo and plot P(team_1 wins):
#   (1) 16-team Swiss (bo1 to 3W/3L) → 8-team single-elim (bo5)
#   (2) 16-team double-elimination (bo5, single GF)
#   (3) 16-team round-robin (bo1; H2H mini-league tiebreak, then coin-flip)
# ------------------------------------------------------------

import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rc('axes', titlesize=16, labelsize=14)     # axes title & label
plt.rc('xtick', labelsize=12)                  # x‐tick labels
plt.rc('ytick', labelsize=12)                  # y‐tick labels
plt.rc('legend', fontsize=12, title_fontsize=13)
plt.rc('figure', titlesize=18)   

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
N_SIMS_PER_POINT = 10000
GLOBAL_SEED       = 2025
FORMAT_SEED_DELTA = 7919
SEEDING           = "random"  # or "fixed"
BRACKET_BEST_OF   = 5

# Strength sweep (you can also sweep directly in probability space)
TEAM1_STRENGTHS = np.geomspace(2.0, 130.0, 15)
TEAM2_STRENGTH  = 9.0
OTHER_STRENGTHS = [7.5, 7.0, 6.8, 6.5, 6.2, 6.0, 5.8, 5.5, 5.2, 5.0, 4.8, 4.5, 4.2, 4.0]

# ──────────────────────────────────────────────────────────────
# Probability helpers
# ──────────────────────────────────────────────────────────────
def build_probability_matrix(teams: Dict[str, float]) -> Dict[Tuple[str, str], float]:
    P = {}
    names = list(teams.keys())
    for a in names:
        for b in names:
            if a == b:
                continue
            sa, sb = teams[a], teams[b]
            P[(a, b)] = 0.5 if sa + sb <= 0 else sa / (sa + sb)
    return P

def play_bo_n(a: str, b: str, P: Dict[Tuple[str, str], float], n: int, rng: random.Random) -> str:
    assert n % 2 == 1
    need = n // 2 + 1
    wa = wb = 0
    p = P[(a, b)]
    while wa < need and wb < need:
        if rng.random() < p:
            wa += 1
        else:
            wb += 1
    return a if wa > wb else b

def play_bo1(a: str, b: str, P: Dict[Tuple[str, str], float], rng: random.Random) -> str:
    return a if rng.random() < P[(a, b)] else b

# ──────────────────────────────────────────────────────────────
# Swiss stage (unchanged from your script)
# ──────────────────────────────────────────────────────────────
def swiss_pairing(buckets, played, rng: random.Random, max_shuffle_attempts: int = 20):
    local = {rec: lst[:] for rec, lst in buckets.items()}
    bucket_keys = sorted(local.keys(), key=lambda x: (-x[0], x[1]))
    pairs = []
    for i, key in enumerate(bucket_keys):
        if len(local[key]) % 2 == 1:
            if i + 1 < len(bucket_keys):
                t = rng.choice(local[key]); local[key].remove(t); local[bucket_keys[i+1]].append(t)
            elif i - 1 >= 0:
                t = rng.choice(local[key]); local[key].remove(t); local[bucket_keys[i-1]].append(t)
    for key in bucket_keys:
        group = local[key]
        if not group:
            continue
        rng.shuffle(group)
        for _ in range(max_shuffle_attempts):
            ok = True
            for i in range(0, len(group) - 1, 2):
                a, b = group[i], group[i+1]
                if b in played[a]:
                    ok = False
                    break
            if ok:
                break
            rng.shuffle(group)
        for i in range(0, len(group) - 1, 2):
            pairs.append((group[i], group[i+1]))
    return pairs

def run_swiss_stage(
    team_names: List[str],
    P: Dict[Tuple[str, str], float],
    rng: random.Random,
    win_target: int = 3,
    loss_limit: int = 3
):
    W = {t: 0 for t in team_names}
    L = {t: 0 for t in team_names}
    alive = set(team_names)
    qualified, eliminated = [], []
    played = {t: set() for t in team_names}

    while len(qualified) < 8:
        buckets = defaultdict(list)
        for t in alive:
            if t not in qualified and t not in eliminated:
                buckets[(W[t], L[t])].append(t)
        pairs = swiss_pairing(buckets, played, rng)

        for a, b in pairs:
            if a not in alive or b not in alive:
                continue
            w = play_bo1(a, b, P, rng)
            l = b if w == a else a
            W[w] += 1
            L[l] += 1
            played[a].add(b); played[b].add(a)

        for t in list(alive):
            if W[t] >= win_target and t not in qualified:
                qualified.append(t); alive.remove(t)
            elif L[t] >= loss_limit and t not in eliminated:
                eliminated.append(t); alive.remove(t)

        if not pairs and len(qualified) >= 8:
            break
        if len(qualified) > 8:
            # simple trim if overshoot (you can swap to the tiebreak Swiss you tested earlier)
            rng.shuffle(qualified)
            qualified = qualified[:8]
            break

    final_records = {t: (W[t], L[t]) for t in team_names}
    for t in team_names:
        if t not in qualified and t not in eliminated:
            eliminated.append(t)
    return qualified, eliminated, final_records

# ──────────────────────────────────────────────────────────────
# 8-team single-elim bracket (bo5)
# ──────────────────────────────────────────────────────────────
def run_bracket(qualifiers: List[str], P, rng, best_of=5, seeding="random") -> str:
    teams = qualifiers[:]
    if seeding == "random":
        rng.shuffle(teams)
    qf_pairs = [(teams[i], teams[i+1]) for i in range(0, 8, 2)]
    sf = [play_bo_n(a, b, P, best_of, rng) for a, b in qf_pairs]
    sf_pairs = [(sf[0], sf[1]), (sf[2], sf[3])]
    f  = [play_bo_n(a, b, P, best_of, rng) for a, b in sf_pairs]
    ch = play_bo_n(f[0], f[1], P, best_of, rng)
    return ch

# ──────────────────────────────────────────────────────────────
# General double-elimination for any team count (multiple of 2)
#   - Pair 0-loss pool and 1-loss pool each round
#   - Eliminate at 2 losses
#   - Single bo5 Grand Final (no reset)
#   - Seeding "random" or "fixed"
# ──────────────────────────────────────────────────────────────
def run_double_elim_general(
    teams_list: List[str],
    P: Dict[Tuple[str, str], float],
    rng: random.Random,
    best_of: int = 5,
    seeding: str = "random"
) -> str:
    assert len(teams_list) >= 2 and len(teams_list) % 2 == 0
    order = teams_list[:]
    if seeding == "random":
        rng.shuffle(order)

    losses = {t: 0 for t in order}
    active = set(order)

    def pair_and_play(pool: List[str], is_lb: bool):
        rng.shuffle(pool)
        winners, losers = [], []
        for i in range(0, len(pool) - 1, 2):
            a, b = pool[i], pool[i+1]
            w = play_bo_n(a, b, P, best_of, rng)
            l = b if w == a else a
            winners.append(w); losers.append(l)
        # update losses/elims
        for l in losers:
            losses[l] += 1
            if losses[l] >= 2:
                active.discard(l)
        return winners, losers

    # loop rounds until only one active or we reach GF
    while True:
        zero = [t for t in active if losses[t] == 0]
        one  = [t for t in active if losses[t] == 1]

        # Grand Final condition
        if len(zero) == 1 and len(one) == 1:
            a, b = zero[0], one[0]
            return play_bo_n(a, b, P, best_of, rng)

        progressed = False

        # Winners (0-loss) round
        if len(zero) >= 2:
            pair_and_play(zero, is_lb=False)
            progressed = True

        # Losers (1-loss) round
        one = [t for t in active if losses[t] == 1]
        if len(one) >= 2:
            pair_and_play(one, is_lb=True)
            progressed = True

        # If no round happened, then either only one team remains or pools are 1+0
        if not progressed:
            if len(active) == 1:
                return next(iter(active))
            # If we have two teams both at 0-loss (rare when N=2 power of two)
            if len(zero) == 2 and len(one) == 0:
                a, b = zero
                return play_bo_n(a, b, P, best_of, rng)
            # If we have two teams both at 1-loss (LB final without WB survivor)
            if len(one) == 2 and len(zero) == 0:
                a, b = one
                return play_bo_n(a, b, P, best_of, rng)
            # Last resort: break a deadlock by promoting to GF
            if len(active) == 2:
                a, b = list(active)
                return play_bo_n(a, b, P, best_of, rng)

# ──────────────────────────────────────────────────────────────
# Round-robin (bo1). H2H mini-league tiebreak, then coin-flip.
# ──────────────────────────────────────────────────────────────
# def run_round_robin_16(
#     team_names: List[str],
#     P: Dict[Tuple[str, str], float],
#     rng: random.Random
# ) -> str:
#     W = {t: 0 for t in team_names}
#     # record results to break ties via H2H mini-league
#     results = {t: {} for t in team_names}

#     for i in range(len(team_names)):
#         for j in range(i+1, len(team_names)):
#             a, b = team_names[i], team_names[j]
#             w = play_bo1(a, b, P, rng)
#             l = b if w == a else a
#             W[w] += 1
#             results[a][b] = 1 if w == a else 0
#             results[b][a] = 1 if w == b else 0

#     maxW = max(W.values())
#     tied = [t for t, w in W.items() if w == maxW]
#     if len(tied) == 1:
#         return tied[0]

#     # Head-to-head mini-league among tied teams
#     h2h = {t: 0 for t in tied}
#     for a in tied:
#         for b in tied:
#             if a == b: continue
#             h2h[a] += results[a].get(b, 0)
#     best_h2h = max(h2h.values())
#     tied_h2h = [t for t in tied if h2h[t] == best_h2h]
#     if len(tied_h2h) == 1:
#         return tied_h2h[0]

#     # final tiebreak: coin-flip among the remaining tied teams
#     return rng.choice(tied_h2h)


def run_round_robin_16(
    team_names: List[str],
    P: Dict[Tuple[str, str], float],
    rng: random.Random,
    rounds: int = 1,
    best_of: int = 1,
) -> str:
    return run_round_robin(team_names, P, rng, rounds=rounds, best_of=best_of)


def run_round_robin(
    team_names: List[str],
    P: Dict[Tuple[str, str], float],
    rng: random.Random,
    rounds: int = 1,         # how many full round-robins (1=single, 2=double, 3=triple…)
    best_of: int = 1,        # bo1 by default; set to an odd number (e.g., 3, 5)
) -> str:
    """
    Round-robin league:
      - Each pair (A,B) plays `rounds` times.
      - Each meeting is a best-of-`best_of` series (odd).
      - League points = number of series won.
      - Ties broken by:
          1) Head-to-head series wins within the tied group (all meetings),
          2) Sonneborn–Berger-like score: sum_over_opponents( series_wins_vs_opp * opp_total_series_wins ),
          3) Random among any remaining exact ties.
    Returns the champion team name.
    """
    assert rounds >= 1
    assert best_of % 2 == 1, "best_of must be odd (1,3,5,...)"

    # league table = series wins
    W = {t: 0 for t in team_names}

    # store series wins per ordered pair for tiebreaks
    h2h_series_wins = {a: {b: 0 for b in team_names if b != a} for a in team_names}

    # play all meetings
    for _ in range(rounds):
        for i in range(len(team_names)):
            for j in range(i + 1, len(team_names)):
                a, b = team_names[i], team_names[j]
                if best_of == 1:
                    winner = play_bo1(a, b, P, rng)
                else:
                    winner = play_bo_n(a, b, P, best_of, rng)
                loser = b if winner == a else a
                W[winner] += 1
                h2h_series_wins[winner][loser] += 1

    # find top by league points
    maxW = max(W.values())
    tied = [t for t, w in W.items() if w == maxW]
    if len(tied) == 1:
        return tied[0]

    # Tiebreak 1: head-to-head series wins within tied group
    h2h_scores = {t: 0 for t in tied}
    for a in tied:
        h2h_scores[a] = sum(h2h_series_wins[a].get(b, 0) for b in tied if b != a)
    best_h2h = max(h2h_scores.values())
    tied_h2h = [t for t in tied if h2h_scores[t] == best_h2h]
    if len(tied_h2h) == 1:
        return tied_h2h[0]

    # Tiebreak 2: Sonneborn–Berger-like (strength of opposition based on league points)
    #   Sum over all opponents: (series_wins_vs_opponent * opponent_total_series_wins)
    sb_score = {}
    for a in tied_h2h:
        s = 0
        for opp in team_names:
            if opp == a:
                continue
            s += h2h_series_wins[a].get(opp, 0) * W[opp]
        sb_score[a] = s
    best_sb = max(sb_score.values())
    tied_sb = [t for t in tied_h2h if sb_score[t] == best_sb]
    if len(tied_sb) == 1:
        return tied_sb[0]

    # Final tiebreak: random among remaining exact ties (very rare with larger rounds/best_of)
    return rng.choice(tied_sb)

# ──────────────────────────────────────────────────────────────
# Format simulators → per-team title probabilities
# ──────────────────────────────────────────────────────────────
def simulate_format_worlds_swiss_se(
    teams: Dict[str, float],
    n_sims: int,
    seed: int,
    bracket_best_of: int = 5,
    bracket_seeding: str = "random",
) -> Dict[str, float]:
    rng = random.Random(seed)
    P = build_probability_matrix(teams)
    names = list(teams.keys())
    wins = Counter()
    for _ in range(n_sims):
        qualifiers, _, _ = run_swiss_stage(names, P, rng, win_target=3, loss_limit=3)
        champion = run_bracket(qualifiers, P, rng, best_of=bracket_best_of, seeding=bracket_seeding)
        wins[champion] += 1
    total = float(n_sims)
    return {t: wins[t] / total for t in names}

def simulate_format_double_elim_16(
    teams: Dict[str, float],
    n_sims: int,
    seed: int,
    best_of: int = 5,
    seeding: str = "random",
) -> Dict[str, float]:
    rng = random.Random(seed)
    P = build_probability_matrix(teams)
    names = list(teams.keys())
    wins = Counter()
    for _ in range(n_sims):
        champ = run_double_elim_general(names, P, rng, best_of=best_of, seeding=seeding)
        wins[champ] += 1
    total = float(n_sims)
    return {t: wins[t] / total for t in names}

def simulate_format_round_robin_16(
    teams: Dict[str, float],
    n_sims: int,
    seed: int,
    rounds: int,
    best_of: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    P = build_probability_matrix(teams)
    names = list(teams.keys())
    wins = Counter()
    for _ in range(n_sims):
        champ = run_round_robin_16(names, P, rng, rounds=rounds, best_of=best_of)
        wins[champ] += 1
    total = float(n_sims)
    return {t: wins[t] / total for t in names}

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def make_teams_dict(team1_strength: float) -> Dict[str, float]:
    s1 = float(max(0.0, team1_strength))
    teams = {"team_1": s1, "team_2": TEAM2_STRENGTH}
    for i, s in enumerate(OTHER_STRENGTHS[:14], start=3):
        teams[f"team_{i}"] = float(max(0.0, s))
    return teams

def head_to_head_win_pct(s1: float, s2: float) -> float:
    s1 = max(0.0, s1); s2 = max(0.0, s2)
    return 50.0 if (s1 + s2) <= 0 else 100.0 * (s1 / (s1 + s2))

# ──────────────────────────────────────────────────────────────
# Main: sweep & plot
# ──────────────────────────────────────────────────────────────
def main():
    x_head2head = []
    y_worlds = []
    y_de16   = []
    y_singlebo1   = []
    y_doublebo1 = []
    y_singlebo5   = []


    for k, s1 in enumerate(TEAM1_STRENGTHS):
        teams = make_teams_dict(s1)
        x_head2head.append(head_to_head_win_pct(s1, TEAM2_STRENGTH))

        seed1 = GLOBAL_SEED + k
        seed2 = GLOBAL_SEED + FORMAT_SEED_DELTA + k
        seed3 = GLOBAL_SEED + 2*FORMAT_SEED_DELTA + k

        probs_worlds = simulate_format_worlds_swiss_se(
            teams, n_sims=N_SIMS_PER_POINT, seed=seed1,
            bracket_best_of=BRACKET_BEST_OF, bracket_seeding=SEEDING
        )
        probs_de16 = simulate_format_double_elim_16(
            teams, n_sims=N_SIMS_PER_POINT, seed=seed2,
            best_of=BRACKET_BEST_OF, seeding=SEEDING
        )
        probs_single_rrbo1 = simulate_format_round_robin_16(
            teams, n_sims=N_SIMS_PER_POINT, seed=seed3, rounds=1, best_of=1
        )

        probs_double_rrbo1 = simulate_format_round_robin_16(
            teams, n_sims=N_SIMS_PER_POINT, seed=seed3, rounds=2, best_of=1
        )

        probs_single_rrbo5 = simulate_format_round_robin_16(
            teams, n_sims=N_SIMS_PER_POINT, seed=seed3, rounds=1, best_of=5
        )

        y_worlds.append(100.0 * probs_worlds.get("team_1", 0.0))
        y_de16.append(100.0 * probs_de16.get("team_1", 0.0))
        y_singlebo1.append(100.0 * probs_single_rrbo1.get("team_1", 0.0))
        y_doublebo1.append(100.0 * probs_double_rrbo1.get("team_1", 0.0))
        y_singlebo5.append(100.0 * probs_single_rrbo5.get("team_1", 0.0))

    # Plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(x_head2head, y_worlds, marker="o", label="Worlds: Swiss → SE (bo5)")
    plt.plot(x_head2head, y_de16,   marker="s", label="Double-elim 16 (bo5, single GF)")
    plt.plot(x_head2head, y_singlebo1,   marker="^", label="Single RR bo1")
    plt.plot(x_head2head, y_doublebo1,   marker="^", label="Double RR bo1")
    plt.plot(x_head2head, y_singlebo5,   marker="^", label="Single RR bo5")
    plt.xlabel("Head-to-head: P(team_1 beats team_2) [%]")
    plt.ylabel("P(team_1 wins tournament) [%]")
    # plt.title("Monte Carlo: team_1 championship probability across three formats")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("5FormatsComparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
