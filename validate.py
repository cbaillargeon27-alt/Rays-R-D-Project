from model import *
rng = np.random.default_rng(7)
full = to_a_scale(load())                     # best-available A-scale value for every ball
full['half'] = rng.integers(0, 2, len(full))
H1, H2 = full[full.half == 0].copy(), full[full.half == 1]
truth = H2.groupby('batter').agg(t=('y', 'mean'), n2=('y', 'size'))

def score(pred, label, subset=None):
    j = truth.join(pred.rename('p'), how='inner').dropna()
    if subset is not None: j = j[j.index.isin(subset)]
    # truth itself is noisy; n2-weighted MSE minus known noise = error in estimating true talent
    mse = np.average((j.t - j.p) ** 2, weights=j.n2)
    print(f'{label:45s} batters={len(j):4d}  wMSE={mse:6.2f}  bias={np.average(j.p - j.t, weights=j.n2):+.2f}')
    return mse

raw = H1.dropna(subset=['speed_A', 'speed_B'], how='all')
print('--- Method comparison (fit on half 1, predict half 2 batter mean, A scale) ---')
gm = H1.y.mean()
score(H1[H1.src == 'A'].groupby('batter').speed_A.mean(), 'Raw mean, system A only (drop non-A)')
score(H1.groupby('batter').y.mean(), 'Raw mean, A + calibrated B')
score(pd.Series(gm, index=truth.index), 'League mean for everyone')
for pit, het in [(False, False), (True, False), (True, True)]:
    f = fit_re(to_a_scale(load().loc[H1.index]), pitcher=pit, hetero=het)
    b = f['batters'].set_index('batter'); pred = f['mu'] + b.effect
    score(pred, f'Random effects, pitcher={pit}, hetero wts={het}')

print('\n--- Scenario tests: half the batters forced to a single system in half 1 ---')
bats = H1.batter.unique(); G = set(rng.choice(bats, len(bats) // 2, replace=False))
base = load().loc[H1.index]
ref = fit_re(to_a_scale(base)); refp = ref['mu'] + ref['batters'].set_index('batter').effect
score(refp, 'Both systems (reference), group G', G)
for scen in ['A_only', 'B_only']:
    x = base.copy(); g = x.batter.isin(G)
    if scen == 'A_only': x.loc[g, ['speed_B', 'vangle_B']] = np.nan
    else: x.loc[g, ['speed_A', 'vangle_A']] = np.nan
    paired = x[~g].dropna(subset=['speed_A', 'speed_B'])       # calibrate only on other batters
    f = fit_re(to_a_scale(x, paired=paired))
    pred = f['mu'] + f['batters'].set_index('batter').effect
    score(pred, f'{scen} for group G (model)', G)
    # naive alternatives for comparison
    if scen == 'A_only':
        score(x[g].groupby('batter').speed_A.mean(), 'A_only naive: raw A mean (drops missing)', G)
    else:
        off = (paired.speed_A - paired.speed_B).mean()
        score(x[g].groupby('batter').speed_B.mean() + off, 'B_only naive: raw B mean + avg offset', G)
