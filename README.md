[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Holladitt/makefun-gta6-trophy-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Holladitt/makefun-gta6-trophy-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

make.fun entry for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

A program (`main.py`, Python 3.13 standard library only) that reads one case
(`/input/<id>.in`) — every Exophase read of the game's trophy page before
the case's checkpoint — and writes `/output/<id>.out`: for each of the game's
55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its
earn rate at the case's target.

## The idea: forecast the counts, not the ratio

An earn rate is a ratio — earners over players tracked — and the house's
recording shows why that matters: on launch night Exophase refreshed its
counts once (638 players tracked became 3,868) and every rate fell, even
though nobody un-earned a trophy. Extrapolating the rate itself chases
refresh artefacts; forecasting the two counts separately does not.

1. **Collapse to refresh points.** A read is kept only when the player count
   or any rate differs from the previous kept read, so long flat stretches
   count as one observation.
2. **Earners and players are cumulative.** Per trophy,
   `earners = rate * players_tracked / 100`; both series are forced monotone
   with a running max to kill refresh noise.
3. **Power-law adoption.** Each series is fitted in log space against
   `log(1 + hours since launch)` by least squares and forecast at the case
   target. The median rate is `100 * earners / players`. A forecast below
   the last observed earners is clamped up to it — cumulative counts never
   fall. Fewer than 3 refresh points (or fewer than 2 valid earner points
   for a trophy) falls back to persistence: the last read's rate.
4. **Honest spread.** The spread comes from the fits' residuals (earners and
   players combined in quadrature), grown with the square root of horizon
   over observed span, with a floor of 0.75 logit units — a flat history
   says nothing about the next refresh, so the 90% band never pretends
   otherwise. Quantiles are taken on the logit scale and rounded to 4
   decimals, half-up.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: the same case always writes the same bytes. No network, no
dependencies beyond the standard library.
