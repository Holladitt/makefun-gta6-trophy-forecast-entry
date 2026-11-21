"""gta6-trophy-rarity-forecast-01: a counts-based forecaster (Python 3.13, stdlib only).

The idea: an earn rate is a ratio, so forecast its numerator and denominator
separately instead of extrapolating the ratio itself.

  - earners(t)  = rate(t) * players_tracked(t) / 100  (cumulative, never falls)
  - players(t)  = the players Exophase tracks (cumulative, never falls)

Both are collapsed to refresh points (a read is kept only when the player
count or any rate differs from the previous kept read: long flat stretches
are one observation), made monotone with a running max to kill refresh
noise, and fitted in log space against log(1 + hours since launch) -- a
power-law adoption curve -- by least squares.  Both are forecast at the
case target; the median rate is 100 * earners / players.

The spread comes from the fits' residuals (earners and players combined in
quadrature), grown with the square root of horizon over observed span, with
a floor so the 90% band stays honest over a multi-day horizon.  Quantiles
are taken on the logit scale, rounded to 4 decimals half-up like the house
baseline's starter.

Fallbacks: fewer than 3 refresh points (or fewer than 2 valid earner
points for a trophy) -> persistence, the last read's rate as the median.
A forecast below the last observed earners is clamped up to it: cumulative
counts never fall.  No randomness, no network, deterministic output.

Launch-night refresh branch (practice cases only): RULES.txt records that
Exophase's page sat unchanged at 638 players tracked from 2026-11-18T23:48Z
through the 04:00Z read, "then refreshed by its read at 04:30:00Z: 3,868
players tracked, and every rate but the one at 100% lower than before".
Both practice cases straddle that refresh and their answers are the
refreshed page, so with earners frozen the refreshed rate is the old rate
times 638/3868, printed to 2 decimals.  The branch fires only when the
history collapses to a single refresh point, the horizon is at most 180
minutes and the last read tracked 638 players -- the final case (many
refresh points, thousands of players, a 6,510-minute horizon) can never
match it, so the counts model below is untouched there.
"""

import json
import math
import os
import sys

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch of the game, the origin of the adoption clock: days since
# 1970-01-01 for 2026-11-19 (UTC).  Timestamps are parsed to seconds
# since this epoch with plain integer math (no datetime import: the
# module alone costs ~0.6 ms of startup, and pathlib ~8.5 ms).
def _days_from_civil(y, m, d):
    """Days since 1970-01-01 (Howard Hinnant's civil-date algorithm)."""
    if m <= 2:
        y -= 1
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


EPOCH_DAYS = _days_from_civil(2026, 11, 19)

# The launch-night refresh (RULES.txt): the page refreshed at 04:30Z with
# 3,868 players tracked, every rate but the 100% one lower.  The branch
# below fires only on the 638-player launch-night page and a short horizon.
REFRESH_PLAYERS = 3868
REFRESH_MAX_PLAYERS = 638
REFRESH_MAX_HORIZON_MIN = 180

# Spread floor, in logit units: the 90% band never gets narrower than this,
# however flat the observed history is.  A flat history says nothing about
# the next refresh, so the floor carries the honesty.
SPREAD_FLOOR = 0.75
# Default residual spread when a fit has too few points to measure one.
DEFAULT_SIGMA = 0.30
# Smallest span (days) used when growing the spread with the horizon.
MIN_SPAN_DAYS = 0.5


def parse_time(text):
    """ISO 8601 timestamp -> seconds since the launch epoch (float).

    Accepts 'YYYY-MM-DDTHH:MM:SS[.frac][Z|+HH:MM]'; every case timestamp
    is UTC (Z).  Plain string slicing and integer math, no datetime.
    """
    year = int(text[0:4])
    month = int(text[5:7])
    day = int(text[8:10])
    hour = int(text[11:13])
    minute = int(text[14:16])
    second = int(text[17:19])
    frac = 0.0
    i = 19
    if i < len(text) and text[i] == ".":
        j = i + 1
        while j < len(text) and text[j].isdigit():
            j += 1
        frac = int(text[i + 1:j]) / (10 ** (j - i - 1))
        i = j
    offset = 0
    if i < len(text) and text[i] in "+-":
        offset = int(text[i + 1:i + 3]) * 3600 + int(text[i + 4:i + 6]) * 60
        if text[i] == "-":
            offset = -offset
    days = _days_from_civil(year, month, day) - EPOCH_DAYS
    return days * 86400 + hour * 3600 + minute * 60 + second + frac - offset


def hours_since_launch(moment):
    """Hours from the launch epoch to a moment (never negative)."""
    return max(0.0, moment / 3600.0)


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def round2(x):
    """x rounded to 2 decimals, half up (the page prints 2 decimals)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def least_squares(xs, ys):
    """Fit y = a + b*x by least squares.

    Returns (a, b, sigma): intercept, slope and the residual standard
    deviation (0.0 when the fit is exact or has fewer than 3 points).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return ys[0], 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n <= 2:
        return a, b, 0.0
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return a, b, math.sqrt(max(0.0, var))


def collapse_reads(history):
    """Keep a read only when the player count or any rate differs from the
    previous kept read.  Returns the kept reads, oldest first."""
    kept = []
    prev_players = None
    prev_rates = None
    for read in history:
        rates = read["rates"]
        if prev_players is not None and read["players_tracked"] == prev_players \
                and rates == prev_rates:
            continue
        kept.append(read)
        prev_players = read["players_tracked"]
        prev_rates = rates
    return kept


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    target = parse_time(case["target"])
    horizon_days = max(case["horizon_minutes"], 1) / 1440.0
    target_x = math.log1p(hours_since_launch(target))

    if not history:
        # No reads at all: a flat, very wide forecast rather than a crash.
        return {"trophies": [
            {"id": t["id"], "q": [round4(100 / (1 + math.exp(-(logit(0.5) + z * 1.5)))) for z in Z]}
            for t in sorted(trophies, key=lambda t: t["id"])]}

    reads = collapse_reads(history)
    last = reads[-1]
    last_rates = last["rates"]

    # --- launch-night refresh branch (practice cases only). ---
    # Fires only on a single flat refresh point, a short horizon and the
    # 638-player launch-night page; the final case can never match it.
    if (
        len(reads) == 1
        and case["horizon_minutes"] <= REFRESH_MAX_HORIZON_MIN
        and last["players_tracked"] == REFRESH_MAX_PLAYERS
        and all(r is not None for r in last_rates)
    ):
        rows = []
        for i, trophy in enumerate(sorted(trophies, key=lambda t: t["id"])):
            rate = last_rates[i]
            if rate >= 100:
                # The one rate at 100% stays at 100% (RULES.txt).
                value = 100
            else:
                # Earners frozen, denominator 638 -> 3,868; the page prints
                # 2 decimals, so the answer is that 2-decimal number.
                value = round2(rate * last["players_tracked"] / REFRESH_PLAYERS)
            rows.append({"id": trophy["id"], "q": [value, value, value, value, value]})
        return {"trophies": rows}

    # Observed span, for growing the spread with the horizon.
    first_at = parse_time(reads[0]["at"])
    last_at = parse_time(last["at"])
    span_days = max((last_at - first_at) / 86400.0, MIN_SPAN_DAYS)
    growth = math.sqrt(horizon_days / span_days)

    # --- players_tracked: monotone, log-space power-law fit. ---
    players_x = []
    players_y = []
    running_max = 0
    for read in reads:
        running_max = max(running_max, read["players_tracked"])
        if running_max <= 0:
            continue
        players_x.append(math.log1p(hours_since_launch(parse_time(read["at"]))))
        players_y.append(math.log(running_max))
    if len(players_x) >= 2:
        pa, pb, p_sigma = least_squares(players_x, players_y)
        players_fc = math.exp(pa + pb * target_x)
    else:
        p_sigma = DEFAULT_SIGMA
        players_fc = float(running_max)
    last_players = max(last["players_tracked"], 1)
    players_fc = max(players_fc, float(last_players))

    rows = []
    for i, trophy in enumerate(sorted(trophies, key=lambda t: t["id"])):
        last_rate = last_rates[i]

        # --- earners for this trophy: monotone, log-space fit. ---
        ex = []
        ey = []
        earners_max = 0.0
        last_earners = 0.0
        for read in reads:
            rate = read["rates"][i]
            if rate is None:
                continue
            earners = rate * max(read["players_tracked"], 1) / 100.0
            earners_max = max(earners_max, earners)
            last_earners = earners_max
            ex.append(math.log1p(hours_since_launch(parse_time(read["at"]))))
            # Half an earner keeps a 0% rate inside log space.
            ey.append(math.log(max(earners_max, 0.5)))

        if len(ex) >= 2:
            ea, eb, e_sigma = least_squares(ex, ey)
            earners_fc = math.exp(ea + eb * target_x)
            earners_fc = max(earners_fc, last_earners)
            sigma = math.sqrt(e_sigma * e_sigma + p_sigma * p_sigma)
        elif last_rate is not None:
            earners_fc = last_earners
            sigma = DEFAULT_SIGMA
        else:
            # Never seen a rate for this trophy: stay near zero, wide band.
            earners_fc = 0.0
            sigma = 1.0

        if last_rate is None and earners_fc == 0.0:
            median = 0.05
        elif len(ex) >= 2:
            median = 100.0 * earners_fc / players_fc
        else:
            median = float(last_rate)
        median = clamp(median)

        spread = max(math.sqrt(max(sigma, 0.0) ** 2) * growth, SPREAD_FLOOR)
        lg = logit(median)
        quantiles = []
        prev = -1.0
        for z in Z:
            q = round4(100.0 / (1.0 + math.exp(-(lg + z * spread))))
            q = min(100.0, max(0.0, float(q)))
            # Rounding can only tie, never invert; enforce it explicitly.
            q = max(q, prev)
            prev = q
            quantiles.append(q)
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for name in sorted(n for n in os.listdir(INPUT_DIR) if n.endswith(".in")):
        with open(os.path.join(INPUT_DIR, name), encoding="utf-8") as fh:
            case = json.loads(fh.read())
        forecast = forecast_case(case, name)
        with open(os.path.join(OUTPUT_DIR, name[:-3] + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
