# Project Write-Up: Projecting Batter Exit Velocity

**Interactive report:** [cbaillargeon27-alt.github.io/Rays-R-D-Project](https://cbaillargeon27-alt.github.io/Rays-R-D-Project/)

## The task

Project each batter's "true" average exit velocity (speed-off-bat) for next season from one season of batted-ball data. Every batted ball can be measured by two tracking systems, A and B, either of which may be missing. The approach had to work for batters with data from only system A or only system B.

**Data:** 73,375 batted balls from 816 batters and 645 pitchers. Each ball has a batter ID, a pitcher ID, a human-coded hit type (ground ball, line drive, fly ball, popup) and speed plus launch angle from each system.

| Coverage | Balls |
| --- | --- |
| Both systems | 64,943 |
| System A only | 860 |
| System B only | 7,030 |
| Neither system | 542 |

## What I did, step by step

### 1. Compared the two measurement systems

Before modeling, I compared A and B on the 64,943 balls both systems measured.

- **In the air, B reads about 4 mph below A.** The spread is tight, about 1 mph. Launch angles agree within about 1 degree.
- **On ground balls, B often fails badly.** The gap is bimodal. About half of ground balls show the normal 4 mph offset; the other half read 10 to 40 mph low on B. On steep grounders (A angle below -10 degrees) the median gap is 25 to 28 mph. B's angle on those balls also reads flatter than A's, which suggests B is picking the ball up after its first bounce.
- **A's weakness is coverage, not accuracy.** A misses 47% of popups and 14% of ground balls. The ground balls A misses average 47 mph on B, against 71 mph for ground balls A captured. A systematically drops weak contact.
- **B's gaps are skewed too.** Ground balls B misses average 75 mph on A, against 86 mph for ground balls B captured.

| A launch angle | Median A − B (mph) |
| --- | --- |
| Below -30 | 28.0 |
| -30 to -20 | 28.0 |
| -20 to -10 | 24.8 |
| -10 to -5 | 12.9 |
| -5 to 0 | 3.5 |
| 0 to 30 | 4.0 to 4.2 |
| 30 to 50 | 3.5 to 3.8 |
| 50+ | 2.9 |

**Takeaway:** a single constant offset between systems would be wrong. Dropping missing readings would also be wrong, because the missing balls are not random.

### 2. Put every ball on system A's scale

I chose system A as the target scale. It agrees with B wherever B is reliable and stays plausible where B breaks down.

| Ball has | Value used |
| --- | --- |
| An A reading (65,803 balls) | A's speed |
| Only a B reading (7,030) | Calibrated prediction of A from B |
| No reading (542) | Average calibrated speed of A-missing balls with the same hit type |

**Calibration model.** A gradient-boosted regression predicts A's speed from B's speed, B's launch angle and hit type, trained on the paired balls. B's angle is the key input: a steep negative B angle flags the bounce-tracking failure, so the model learns when to add about 25 mph instead of about 4.

I validated it with 5-fold cross-validation grouped by batter, so no batter's balls were in both training and test:

| Approach | RMSE vs A (mph) |
| --- | --- |
| B plus a constant offset | 10.8 |
| Gradient-boosted calibration | 4.7 |

Remaining error is about 1.1 mph on fly balls and line drives, 1.4 on popups and 7.0 on ground balls.

I also checked that calibration keeps batter-level differences. For the 370 batters with at least 50 paired balls, the batter average of calibrated-from-B speeds correlated 0.96 with the batter average of A speeds (slope 0.95).

**Why impute instead of dropping missing balls?** The balls A misses are weak contact. Dropping them makes A-only averages about 1.5 mph too high, and the error is worst for batters who make a lot of weak contact.

### 3. Built the projection model

Each batter's projection comes from a crossed random-effects model fit by EM:

```
speed = league mean + batter effect + pitcher effect + noise
```

| Fitted component | Value (mph) |
| --- | --- |
| League mean (A scale) | 86.2 |
| SD of batter true talent | 3.0 |
| SD of pitcher effect | 1.1 |
| SD of ball-to-ball noise | 13.6 |

- **Shrinkage.** A batter's own average gets weight n / (n + 20.3), so about 20 batted balls gives his data half the weight. Small samples are pulled toward the league mean.
- **Pitcher adjustment.** The projection removes the quality of pitchers each batter happened to face, so it's pitcher-neutral for next season.
- **Equal weighting.** I tested weighting balls by how precisely they were measured. It pushed projections up 0.25 mph, because the imprecise balls (B-only and imputed) are the weak-contact ones. I switched to equal weights, which removed the bias.
- **Next season.** Exit velocity is a stable skill, so the shrunken true-talent estimate is the projection. The data has no ages, so no aging curve is applied.

### 4. Validated the approach

**Split-half test.** I randomly split each batter's balls into two halves, fit every method on half 1 and scored how well it predicted the batter's half-2 average, across 720 batters. Half-2 sampling noise was subtracted to estimate error against true talent.

| Method | Talent RMSE (mph) | Bias (mph) |
| --- | --- | --- |
| League mean for everyone | 2.83 | 0.00 |
| Raw average, system A only | 2.44 | +1.50 |
| Raw average, A + calibrated B | 1.89 | 0.00 |
| Random effects, batter only | 1.38 | -0.01 |
| **Random effects, batter + pitcher (final)** | **1.39** | **-0.01** |
| Random effects, precision-weighted | 1.40 | +0.24 |

**Single-system test.** To confirm the approach works for batters with only one system, I stripped half the batters down to A only, or B only, in half 1. The calibration was retrained without those batters, so it never saw them. The numbers are weighted MSE against half-2 averages for the same 358 batters; lower is better.

| Data available | Model | Naive approach |
| --- | --- | --- |
| Both systems | 5.95 | — |
| A only | 6.19 | 10.03 (raw A average, bias +1.39) |
| B only | 5.67 | 13.15 (B average + flat offset, bias -0.48) |

The model performs about as well with one system as with both. The naive approaches fail in opposite directions: raw A overrates batters because it loses their weak contact, and a flat B offset misprices batters according to how many ground balls they hit.

## Results

- Projections range from **77.9 to 92.8 mph** around a league mean of 86.2 mph, with an SD of 2.3 mph.
- **Batter 47 projects highest** at 92.8 mph (380 batted balls; 80% interval 91.9 to 93.6).
- **Batter 22 projects lowest** at 77.9 mph (36 batted balls; 80% interval 75.5 to 80.2).
- **Shrinkage example:** batter 257 had the highest raw average (95.9 mph) but only 41 balls, so he projects at 92.5.
- 799 batters have data from both systems, 15 have B only (each with 1 or 2 balls) and 2 have no sensor readings. Thin-sample batters project near the league mean with wide intervals.
- To convert a projection to B's scale for balls in the air, subtract about 4 mph. B has no stable scale for ground balls.

The full projections table, with sorting and filters, is in the [interactive report](https://cbaillargeon27-alt.github.io/Rays-R-D-Project/).

## Limitations and next steps

- **Calibrating balls A missed is an extrapolation.** The calibration is trained on balls both systems saw, then applied to the weaker balls A dropped. B's own speed and angle account for most of that difference, but it can't be checked directly.
- **No dates, parks or ages.** With game dates I would weight recent balls more and test year-to-year stability. With ages I would add an aging curve.
- **Average exit velocity mixes skill and approach.** Batters who hit lots of popups score low. 90th-percentile exit velocity, or average exit velocity on balls in the air, would be worth projecting alongside it as purer measures of contact quality.
- **Hit types are human-coded,** so a few may be wrong. The calibration also uses B's launch angle, so it doesn't depend on hit type alone.

## Code

| File | Purpose |
| --- | --- |
| `model.py` | Loads data, calibrates B to A, fits the random-effects model |
| `validate.py` | Produces every validation table above |
| `project.py` | Fits the final model and writes the projections CSV |
| `index.html` | Self-contained interactive report |

**To reproduce:** place `battedBallData.csv` in the repo folder (it isn't included here), install `pandas`, `numpy`, `scipy` and `scikit-learn`, then run `python project.py` and `python validate.py`.
