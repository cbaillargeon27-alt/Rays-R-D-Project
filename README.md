# Batter Exit Velocity Projections

Projects each batter's next-season true average exit velocity from batted-ball data measured by two tracking systems (A and B).

**View the report:** https://cbaillargeon27-alt.github.io/Rays-R-D-Project/

## Approach
1. Compare systems A and B. B reads ~4 mph below A in the air but 25–30 mph low on many ground balls; A misses weak contact.
2. Put every ball on system A's scale: A when measured, a gradient-boosted calibration from B otherwise, a hit-type imputation when neither system measured it.
3. Fit a crossed batter + pitcher random-effects model (EM) and shrink each batter toward the league mean.
4. Validate on split halves and on forced A-only / B-only scenarios.

## Files
- `index.html`: full report (self-contained)
- `model.py`: calibration and random-effects model
- `validate.py`: validation tables
- `project.py`: writes the projections

The input dataset is not included in this repository.
