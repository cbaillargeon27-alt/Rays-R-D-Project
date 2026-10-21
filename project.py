from model import *
d = to_a_scale(load())
f = fit_re(d, pitcher=True)
mu = f['mu']
b = f['batters'].set_index('batter')

g = d.groupby('batter')
out = pd.DataFrame({
    'n_batted_balls': g.size(),
    'n_system_A': g.speed_A.count(),
    'n_system_B': g.speed_B.count(),
    'n_B_only': g.src.apply(lambda s: (s == 'B').sum()),
    'n_neither': g.src.apply(lambda s: (s == 'none').sum()),
    'raw_mean_A': g.speed_A.mean(),
    'raw_mean_calibrated': g.y.mean(),
})
out['data_source'] = np.select(
    [(out.n_system_A > 0) & (out.n_system_B > 0), out.n_system_A > 0, out.n_system_B > 0],
    ['A and B', 'A only', 'B only'], 'no measurements')
out['projected_ev_mph'] = mu + b.effect
out['se_mph'] = b.se
out['lo80'] = out.projected_ev_mph - 1.2816 * out.se_mph
out['hi80'] = out.projected_ev_mph + 1.2816 * out.se_mph
out['shrinkage_weight'] = 1 - (b.se ** 2 / f['vb'])   # share of the batter's own signal kept
out = out.sort_values('projected_ev_mph', ascending=False)
out.insert(0, 'rank', np.arange(1, len(out) + 1))
out = out.round(2).reset_index()
out.to_csv('ev_projections.csv', index=False)

print(f"mu={mu:.2f}  ball sd={f['s2']**.5:.2f}  batter sd={f['vb']**.5:.2f}  pitcher sd={f['vp']**.5:.2f}")
print('balls for 50% reliability:', round(f['s2'] / f['vb'], 1))
print(out.data_source.value_counts())
print(out.head(12).to_string()); print(out.tail(5).to_string())
print(out[out.data_source != 'A and B'].head(8).to_string())
print('proj sd', out.projected_ev_mph.std().round(2), 'range', out.projected_ev_mph.min(), out.projected_ev_mph.max())
