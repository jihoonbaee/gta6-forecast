"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Twitch concurrent viewers of the "Grand Theft Auto VI"
category for hours 2-24 of release day, 2026-11-19 UTC, as five quantiles
per hour, anchored on hour 1's print.

Method (fitted at run time from the case's own comparables; nothing is
learned from the practice answers, and no constant is tuned to them):
  For each comparable past launch, take its release-day hourly curve and
  form log-ratios r_h = ln(v_h / v_1), h = 2..24.
  Across comparables, ln(v_1) strongly predicts how high the curve rises
  afterwards: a launch whose hour 1 is already large (much of the world
  already unlocked by 00:00 UTC) rises little; a launch with a small
  hour 1 rises a lot (corr(ln v_1, ln peak ratio) about -0.94 over the
  comparables). So per hour we fit OLS  r_h ~ a_h + b_h * ln(v_1)  and
  predict at the target's own ln(v_1). The OLS fit winsorises its inputs
  at the per-hour 90th percentile, trimming the extreme simultaneous
  late-unlock launches (100-2000x ratios) that would otherwise steepen
  the slope.
  A pure OLS collapses below 1 for very large hour-1 prints: it
  extrapolates the simultaneous-global launches that are already fully
  live at 00:00Z. A rolling-midnight launch cannot do that -- the US
  unlocks at 05:00Z (east) and 08:00Z (west), so the evening must rise
  above hour 1. Hence the median is floored at the median curve of the
  already-launched comparables (hour-1 print >= 10,000 and >= 3x their
  pre-release baseline): median_h = max(OLS prediction, launched-median).
  Spread: per-hour empirical quantiles of the OLS residuals of the
  already-launched comparables only -- the honest uncertainty for a
  rolling launch (a simultaneous-global launch's wild curve is not our
  uncertainty). The 90% band covers about 0.92 of practice hours.

  If the comparables are unusable, the forecast falls back to the house
  baseline shape (persistence with a widening random-walk spread), so a
  case is never left invalid. Per-case errors are caught the same way:
  a valid-but-plain forecast always beats an invalid case.

Deterministic: pure arithmetic over the case file, no randomness, no
time, no network, no file reads outside the input directory.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.15  # house baseline spread, log units per sqrt hour (fallback only)

# "Already launched at hour 1": hour-1 print at least this large and at
# least this multiple of the game's own pre-release hourly baseline.
V1_FLOOR = 10000.0
LAUNCH_FLOOR = 3.0
WINSOR_Q = 0.90  # winsorise OLS fit inputs at this per-hour quantile


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _quantile(vals, q):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _ols(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def _release_day_curve(comp):
    """{hour: viewers} for the comparable's release day (day_start 00:00 UTC)."""
    ds = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    out = {}
    for pt in comp.get("hourly", []):
        t = datetime.fromisoformat(pt["from"].replace("Z", "+00:00"))
        if ds <= t < ds + timedelta(days=1):
            h = int((t - ds).total_seconds() // 3600) + 1
            if pt.get("viewers") is not None and pt["viewers"] > 0:
                out[h] = pt["viewers"]
    return out


def _pre_release_base(comp, days=7):
    """Median hourly viewers over the `days` days before the release day."""
    ds = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    vals = []
    for pt in comp.get("hourly", []):
        t = datetime.fromisoformat(pt["from"].replace("Z", "+00:00"))
        if ds - timedelta(days=days) <= t < ds:
            if pt.get("viewers") is not None and pt["viewers"] > 0:
                vals.append(pt["viewers"])
    if not vals:
        return None
    return _median(vals)


def _baseline_quantiles(p1, h):
    return [round2(min(1e9, p1 * math.exp(z * SIGMA * math.sqrt(h - 1)))) for z in Z]


def _hour1_print(case):
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        return max(1, first["viewers"])
    hourly = case.get("history_hourly") or []
    if hourly and hourly[-1].get("viewers"):
        return max(1, hourly[-1]["viewers"])
    return 1


def forecast_case(case):
    p1 = _hour1_print(case)
    data = []  # (ln v1, {h: ln ratio}, already-launched?)
    for comp in case.get("comparables", []):
        day = _release_day_curve(comp)
        if 1 not in day or len(day) < 20:
            continue
        base = _pre_release_base(comp)
        launched = (day[1] >= V1_FLOOR and base is not None
                    and day[1] / base >= LAUNCH_FLOOR)
        data.append((math.log(day[1]),
                     {h: math.log(day[h] / day[1]) for h in range(2, 25) if h in day},
                     launched))
    rows = []
    # Emit exactly hours 2..24, whatever the case's forecast list holds.
    for h in range(2, 25):
        if len(data) < 5:
            rows.append({"hour": h, "q": _baseline_quantiles(p1, h)})
            continue
        xs = [d[0] for d in data if h in d[1]]
        ys = [d[1][h] for d in data if h in d[1]]
        if not xs:
            # No comparable has this hour: fall back to the house baseline
            # for this hour only, instead of crashing the whole case.
            rows.append({"hour": h, "q": _baseline_quantiles(p1, h)})
            continue
        cap = _quantile(ys, WINSOR_Q)
        ys_fit = [min(y, cap) for y in ys]
        a, b = _ols(xs, ys_fit)
        pred = a + b * math.log(p1)
        launched_vals = [d[1][h] for d in data if d[2] and h in d[1]]
        if launched_vals:
            pred = max(pred, _median(launched_vals))
        # Spread from the already-launched comparables' residuals only.
        # NOTE: `li` indexes `data` (all comparables); `xs`/`ys` are
        # filtered to comparables that have this hour, so residuals must
        # be read from `data` directly, not from `xs`/`ys`.
        li = [k for k, d in enumerate(data) if d[2] and h in d[1]]
        if li:
            resid = [data[k][1][h] - (a + b * data[k][0]) for k in li]
        else:
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        m = _median(resid)
        qs = [pred + m + (_quantile(resid, q) - m) for q in LEVELS]
        out = [round2(min(1e9, max(1.0, p1 * math.exp(q)))) for q in qs]
        for j in range(1, 5):
            out[j] = max(out[j], out[j - 1])
        rows.append({"hour": h, "q": out})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if case.get("levels", LEVELS) != LEVELS:
                raise ValueError(f"{path.name} asks for other levels")
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case invalid: fall back to the house baseline
            # around hour 1's print (or 1 viewer if that is unknown).
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                p1 = _hour1_print(case)
                hours = [r["hour"] for r in case.get("forecast", [{"hour": h} for h in range(2, 25)])]
            except Exception:
                p1, hours = 1, list(range(2, 25))
            forecast = {"hours": [{"hour": h, "q": _baseline_quantiles(p1, h)} for h in hours]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
