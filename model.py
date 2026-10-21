"""Batter exit-velocity projection.
Step 1: put every batted ball on system A's scale (A if measured; else calibrated from B; else imputed).
Step 2: crossed random-effects model  y = mu + batter + pitcher + e, with per-ball noise variance
        that reflects how the value was obtained, fit by EM. Projection = mu + batter effect.
"""
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import scipy.sparse as sp

HT = {'ground_ball': 0, 'line_drive': 1, 'fly_ball': 2, 'popup': 3, 'U': 1}

def load(path='battedBallData.csv'):
    d = pd.read_csv(path, na_values=['NA'])
    d['ht'] = d.hittype.map(HT)
    return d

def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                         categorical_features=[2], random_state=0)

def to_a_scale(d, paired=None):
    """Return y (A scale) and per-ball measurement-error variance for each row of d.
    `paired` = rows used to train the B->A calibration (defaults to d's own paired rows)."""
    d = d.copy()
    if paired is None:
        paired = d.dropna(subset=['speed_A', 'speed_B'])
    F = ['speed_B', 'vangle_B', 'ht']
    m = gbm().fit(paired[F].values, paired.speed_A.values)
    # residual variance of the calibration, by hit type and B angle band (ground balls are the problem)
    pr = paired.copy(); pr['res'] = pr.speed_A - m.predict(pr[F].values)  # in-sample; inflate below
    def band(df): return df.ht.astype(str) + '_' + pd.cut(df.vangle_B, [-200, -10, 0, 200], labels=['lo', 'mid', 'hi']).astype(str)
    pr['band'] = band(pr)
    resvar = (pr.groupby('band').res.var() * 1.15).to_dict()   # ~15% inflation for in-sample optimism

    hasA, hasB = d.speed_A.notna(), d.speed_B.notna()
    d['y'] = d.speed_A
    d['mvar'] = 0.0
    d['src'] = np.where(hasA, 'A', np.where(hasB, 'B', 'none'))
    bo = ~hasA & hasB
    d.loc[bo, 'y'] = m.predict(d.loc[bo, F].values)
    d.loc[bo, 'mvar'] = band(d.loc[bo]).map(resvar).fillna(pr.res.var()).values
    # neither system: impute the typical A-missing ball of that hit type (A drops weak contact)
    weak = d.loc[bo].groupby('ht').y.agg(['mean', 'var'])
    nn = ~hasA & ~hasB
    d.loc[nn, 'y'] = d.loc[nn, 'ht'].map(weak['mean']).fillna(d.y.mean()).values
    d.loc[nn, 'mvar'] = d.loc[nn, 'ht'].map(weak['var']).fillna(d.y.var()).values
    return d

def fit_re(d, n_iter=60, verbose=False, pitcher=True, hetero=False):
    """EM for y = mu + b[batter] + p[pitcher] + e, Var(e_i) = s2 + mvar_i."""
    bi, bat = pd.factorize(d.batter); pi, pit = pd.factorize(d.pitcher)
    nb, npt, n = len(bat), len(pit), len(d)
    Z = sp.hstack([sp.csr_matrix((np.ones(n), (np.arange(n), bi)), shape=(n, nb)),
                   sp.csr_matrix((np.ones(n), (np.arange(n), pi)), shape=(n, npt))]).tocsr()
    y = d.y.values
    # Equal weights by default: imputed balls are systematically weak contact, so down-weighting
    # them biases batter means upward. Calibration noise is absorbed into the shared residual variance.
    mv = d.mvar.values if hetero else np.zeros(n)
    mu, s2, vb, vp = y.mean(), y.var() * 0.85, 4.0, 1.0
    for it in range(n_iter):
        w = 1.0 / (s2 + mv)
        W = sp.diags(w)
        A = (Z.T @ W @ Z).toarray()
        A[np.arange(nb), np.arange(nb)] += 1 / vb
        A[nb + np.arange(npt), nb + np.arange(npt)] += 1 / vp
        C = np.linalg.inv(A)
        u = C @ (Z.T @ (w * (y - mu)))
        mu = np.sum(w * (y - Z @ u)) / w.sum()
        cd = np.diag(C)
        vb = (np.sum(u[:nb] ** 2) + cd[:nb].sum()) / nb
        vp = (np.sum(u[nb:] ** 2) + cd[nb:].sum()) / npt if pitcher else 1e-6
        # update s2 (EM step for the shared part of the residual variance)
        r = y - mu - Z @ u
        j, k = bi, nb + pi
        q = cd[j] + cd[k] + 2 * C[j, k]                     # diag(Z C Z'), one batter + one pitcher per row
        frac = s2 / (s2 + mv)
        s2 = np.sum(frac ** 2 * (r ** 2 + q) + s2 * (1 - frac)) / n
    out_b = pd.DataFrame({'batter': bat, 'effect': u[:nb], 'se': np.sqrt(cd[:nb])})
    out_p = pd.DataFrame({'pitcher': pit, 'effect': u[nb:], 'se': np.sqrt(cd[nb:])})
    return dict(mu=mu, s2=s2, vb=vb, vp=vp, batters=out_b, pitchers=out_p)
