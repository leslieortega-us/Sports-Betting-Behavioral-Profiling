"""Reproduce the final descriptive analysis: python final_analysis.py."""
from pathlib import Path
from itertools import combinations
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.cluster.hierarchy import linkage, cut_tree
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'final'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 42]


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight')
    plt.close(fig)


def canonical(labels, features):
    """Label groups by increasing mean stake; labels have no inherent meaning."""
    order = features.assign(cluster=labels).groupby('cluster').average_stake.mean().sort_values().index
    return np.array([dict(zip(order, range(len(order))))[v] for v in labels])


def score(X, labels):
    return float(silhouette_score(X, labels, sample_size=2000, random_state=42))


def main():
    df = pd.read_csv(ROOT / 'data' / 'bets.csv', sep=';')
    checks = {
        'rows': len(df), 'bettors': df.user_id.nunique(),
        'missing_cells': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_bet_ids': int(df.bet_id.duplicated().sum()),
        'invalid_stakes': int((df.stake <= 0).sum()),
        'invalid_odds': int((df.odds <= 1).sum()),
        'invalid_win_values': int((~df.is_win.isin([0, 1])).sum()),
        'payout_mismatches': int((~np.isclose(df.gain, np.where(df.is_win == 1, df.stake * df.odds, 0), atol=.011, rtol=0)).sum()),
        'ggr_mismatches': int((~np.isclose(df.GGR, df.stake - df.gain, atol=.011, rtol=0)).sum()),
    }
    assert all(checks[k] == 0 for k in checks if k not in ['rows', 'bettors']), checks
    (OUT / 'data_checks.json').write_text(json.dumps(checks, indent=2))
    f = df.groupby('user_id').agg(
        total_bets=('bet_id','count'), average_stake=('stake','mean'),
        median_stake=('stake','median'), stake_std=('stake','std'),
        average_odds=('odds','mean'), odds_variance=('odds','var'),
        sport_diversity=('sport','nunique'), bet_type_diversity=('bet_type','nunique'))
    for category in ['sport', 'bet_type']:
        f = f.join(pd.crosstab(df.user_id, df[category], normalize='index').add_prefix(category + '_share_'))
    f['stake_cv'] = f.stake_std / f.average_stake
    model_features = f.drop(columns=['median_stake','stake_std','bet_type_share_single','bet_type_diversity']).copy()
    model_features[['average_stake','odds_variance']] = np.log1p(model_features[['average_stake','odds_variance']])
    assert model_features.shape == (5000,20)
    X = StandardScaler().fit_transform(model_features)
    assert np.isfinite(X).all()
    print('Validated data and created feature matrix', flush=True)

    models, comparison = {}, []
    for k in range(2,9):
        m = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
        labels = canonical(m.labels_, f)
        models[k] = labels
        comparison.append(dict(method='K-Means', k=k, silhouette=score(X, labels), inertia=m.inertia_, smallest_cluster=int(np.bincount(labels).min())))
    baseline = models[2]
    # Ward linkage uses Euclidean distances in exactly the same standardized space.
    tree = linkage(X, method='ward', metric='euclidean')
    cuts = cut_tree(tree, n_clusters=list(range(2,9)))
    for j,k in enumerate(range(2,9)):
        labels = canonical(cuts[:, j], f)
        comparison.append(dict(method='Ward agglomerative', k=k, silhouette=score(X, labels), inertia=np.nan, smallest_cluster=int(np.bincount(labels).min()), ari_to_kmeans_same_k=adjusted_rand_score(models[k],labels)))
    ward2 = canonical(cuts[:,0], f)
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(OUT / 'method_comparison.csv', index=False)
    print(comparison.round(4).to_string(index=False), flush=True)
    stability_labels = {s: KMeans(n_clusters=2, n_init=50, random_state=s).fit_predict(X) for s in SEEDS}
    stability = pd.DataFrame([dict(seed_a=a,seed_b=b,ARI=adjusted_rand_score(stability_labels[a],stability_labels[b])) for a,b in combinations(SEEDS,2)])
    stability.to_csv(OUT / 'seed_stability.csv', index=False)

    # These are sensitivity checks, not a search for the largest silhouette.
    # Scores in different feature spaces do not measure the same geometry.
    sport_idx = [i for i,c in enumerate(model_features) if c.startswith('sport_share_')]
    balanced = X.copy()
    balanced[:,sport_idx] /= np.sqrt(len(sport_idx))
    no_stake = model_features.drop(columns=['average_stake','stake_cv'])
    variants = {
        'baseline_standard_log': X,
        'robust_log': RobustScaler().fit_transform(model_features),
        'sport_block_downweighted': balanced,
        'no_stake_features': StandardScaler().fit_transform(no_stake),
        'standard_no_log': StandardScaler().fit_transform(f[model_features.columns]),
    }
    sensitivity = []
    for name, data in variants.items():
        lab = KMeans(n_clusters=2,n_init=50,random_state=42).fit_predict(data)
        sensitivity.append(dict(variant=name,n_features=data.shape[1],silhouette_own_space=score(data,lab),ari_to_baseline=adjusted_rand_score(baseline,lab),smallest_cluster=int(np.bincount(lab).min())))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / 'feature_sensitivity.csv', index=False)
    print('Completed sensitivity checks', flush=True)

    df['net_profit'] = df.gain-df.stake
    p = df.groupby('user_id').agg(total_bets=('bet_id','size'),wins=('is_win','sum'),total_stake=('stake','sum'),gross_payout=('gain','sum'),net_profit=('net_profit','sum'))
    p['win_rate'] = p.wins / p.total_bets
    p['average_net_profit_per_bet'] = p.net_profit / p.total_bets
    p['roi'] = p.net_profit / p.total_stake
    assert np.allclose(p.net_profit, -df.groupby('user_id').GGR.sum(), atol=.011)
    p['cluster'] = baseline
    p['profile'] = np.where(baseline == 0, 'Lower average stake', 'Higher average stake')
    p['ward_cluster'] = ward2
    p.to_csv(OUT / 'bettor_performance.csv')
    f.assign(cluster=baseline,ward_cluster=ward2).to_csv(OUT / 'bettor_features_and_clusters.csv')
    profiles = f.assign(cluster=baseline).groupby('cluster').mean()
    profiles.insert(0,'bettors', np.bincount(baseline))
    profiles.to_csv(OUT / 'behavior_profiles.csv')
    ward_profiles = f.assign(ward_cluster=ward2).groupby('ward_cluster').mean()
    ward_profiles.insert(0, 'bettors', np.bincount(ward2))
    ward_profiles.to_csv(OUT / 'ward_behavior_profiles.csv')
    assert (f.loc[ward2 == 0, 'sport_share_Cycling'] > 0).all(), 'Recheck Ward cycling interpretation'
    pd.crosstab(pd.Series(baseline,name='kmeans_cluster'),pd.Series(ward2,name='ward_cluster')).to_csv(OUT / 'method_cross_tab.csv')
    summaries=[]
    for c,g in p.groupby('cluster'):
        row=dict(cluster=int(c),bettors=len(g),bets=int(g.total_bets.sum()),mean_win_rate=g.win_rate.mean(),median_win_rate=g.win_rate.median(),mean_net_profit=g.net_profit.mean(),median_net_profit=g.net_profit.median(),mean_profit_per_bet=g.average_net_profit_per_bet.mean(),mean_bettor_roi=g.roi.mean(),median_bettor_roi=g.roi.median(),pooled_roi=g.net_profit.sum()/g.total_stake.sum(),profitable_bettor_share=(g.net_profit>0).mean())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT / 'cluster_performance.csv',index=False)
    # Bettors, not individual bets, are the resampling unit. Fixed memberships.
    # Conditional descriptive bootstrap: does not quantify clustering uncertainty.
    rng=np.random.default_rng(42)
    contrasts=[]
    metrics=['win_rate','net_profit','average_net_profit_per_bet','roi']
    for metric in metrics:
        a=p.loc[p.cluster==0,metric].to_numpy()
        b=p.loc[p.cluster==1,metric].to_numpy()
        differences=np.empty(3000)
        for i in range(len(differences)):
            differences[i]=rng.choice(b,len(b),replace=True).mean()-rng.choice(a,len(a),replace=True).mean()
        lo,hi=np.quantile(differences,[.025,.975])
        contrasts.append(dict(metric=metric,mean_lower_stake=a.mean(),mean_higher_stake=b.mean(),difference_higher_minus_lower=b.mean()-a.mean(),ci95_low=lo,ci95_high=hi,bootstrap_replicates=len(differences)))
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(OUT / 'performance_contrasts.csv',index=False)
    print(pd.DataFrame(summaries).round(4).to_string(index=False), flush=True)
    print(contrasts.round(4).to_string(index=False), flush=True)

    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,g in comparison.groupby('method'):
        ax.plot(g.k,g.silhouette,marker='o',label=name)
    ax.set(xlabel='Number of clusters',ylabel='Sampled silhouette (same 2,000 bettors)',title='Cluster separation in the baseline feature space')
    ax.legend();savefig(fig,'method_comparison.png')
    pca=PCA(n_components=2,svd_solver='full').fit(X)
    coords=pca.transform(X)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True,sharey=True)
    for ax,labels,title in zip(axes,[baseline,ward2],['K-Means (k=2)','Ward agglomerative (k=2)']):
        for c in [0,1]:
            ax.scatter(coords[labels==c,0],coords[labels==c,1],s=5,alpha=.3,label=f'Group {c}')
        ax.set(title=title,xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.legend(markerscale=3)
    savefig(fig,'method_pca.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric,title in zip(axes,['average_stake','stake_cv'],['Mean stake per bettor','Relative stake variability']):
        ax.boxplot([f.loc[baseline==c,metric] for c in [0,1]],tick_labels=['Lower stake','Higher stake'],showfliers=False)
        ax.set(title=title,ylabel='Stake units' if metric=='average_stake' else 'SD / mean')
    savefig(fig,'stake_profiles.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric,title in zip(axes,['win_rate','roi'],['Per-bettor win rate','Per-bettor ROI']):
        ax.boxplot([100*p.loc[p.cluster==c,metric] for c in [0,1]],tick_labels=['Lower stake','Higher stake'],showfliers=False)
        ax.set(title=title,ylabel='Percent')
        if metric=='roi':ax.axhline(0,color='grey',linewidth=1,linestyle='--')
    savefig(fig,'performance_distributions.png')
    metadata=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,scipy=scipy.__version__,scikit_learn=sklearn.__version__,matplotlib=matplotlib.__version__,silhouette_sample_size=2000,random_seed=42,n_init=50,bootstrap_replicates=3000,pca_variance_shown=float(pca.explained_variance_ratio_.sum()),baseline_ward_ari=float(adjusted_rand_score(baseline,ward2)),minimum_seed_ari=float(stability.ARI.min()),maximum_seed_ari=float(stability.ARI.max()))
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Saved results to',OUT,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
