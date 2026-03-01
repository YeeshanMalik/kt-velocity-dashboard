"""
irt_fitting.py — Proper 2PL IRT fitting + EAP ability estimation.

Tiered approach:
  - Tier 1 (≥30 responses): EM-MML 2PL (Bock & Aitkin 1981)
  - Tier 2 (5-29 responses): Lord's approximation from point-biserial r
  - Tier 3 (<5 responses): Population prior (a=1.0, b=0.0)

Also provides running EAP theta (replaces logit(accuracy) proxy).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split


SEED = 42
MIN_RESPONSES_2PL = 30   # Minimum for proper 2PL fitting
MIN_RESPONSES_LORD = 5   # Minimum for Lord's approximation


def get_train_test_split(df):
    """Reproduce the exact train/test split from kt_969_train.py."""
    user_stats = (df.groupby('user_id')
                  .agg(mean_accuracy=('is_correct', 'mean'))
                  .reset_index())
    user_stats['accuracy_bin'] = pd.qcut(
        user_stats['mean_accuracy'], q=4, labels=False, duplicates='drop')
    _tr, _te = train_test_split(
        user_stats.index, test_size=0.2, random_state=SEED,
        stratify=user_stats['accuracy_bin'])
    train_ids = set(user_stats.iloc[_tr]['user_id'])
    test_ids = set(user_stats.iloc[_te]['user_id'])
    return train_ids, test_ids


def _fit_2pl_em_mml(response_matrix, n_quad=21, max_iter=50, tol=1e-3, verbose=True):
    """
    Fit 2PL IRT via Marginal Maximum Likelihood (Bock & Aitkin 1981).

    Vectorized EM algorithm:
      E-step: compute posterior P(θ_k | responses_j) at quadrature points
      M-step: maximize expected marginal log-likelihood for each item's (a_i, b_i)

    Args:
        response_matrix: (n_items, n_students), values 0.0/1.0/NaN
        n_quad: number of quadrature points
        max_iter: maximum EM iterations
        tol: convergence tolerance (max parameter change)

    Returns: (discrimination, difficulty) arrays of shape (n_items,)
    """
    n_items, n_students = response_matrix.shape

    # Quadrature grid with N(0,1) weights
    theta = np.linspace(-4, 4, n_quad)
    log_weights = -0.5 * theta**2 - 0.5 * np.log(2 * np.pi)

    # Initialize parameters
    a = np.ones(n_items)
    b = np.zeros(n_items)
    for i in range(n_items):
        p = np.nanmean(response_matrix[i])
        if np.isfinite(p) and 0.01 < p < 0.99:
            b[i] = -np.log(p / (1 - p))

    # Precompute masks
    valid_masks = ~np.isnan(response_matrix)           # (items, students)
    resp_filled = np.nan_to_num(response_matrix, nan=0.0)  # NaN→0 for matmul

    for iteration in range(max_iter):
        a_old, b_old = a.copy(), b.copy()

        # ── E-step: vectorized posterior computation ──
        # P(correct_ij | theta_k) for all items, quad points
        p_all = expit(a[:, None] * (theta[None, :] - b[:, None]))  # (items, quad)
        p_all = np.clip(p_all, 1e-10, 1 - 1e-10)
        log_p = np.log(p_all)
        log_1mp = np.log(1 - p_all)

        # log P(responses_j | theta_k) = sum_i valid[i,j] * [r[i,j]*log_p[i,k] + (1-r[i,j])*log_1mp[i,k]]
        # Vectorized: (students, quad) = (valid*resp).T @ log_p + (valid*(1-resp)).T @ log_1mp
        log_lik = (valid_masks * resp_filled).T @ log_p + \
                  (valid_masks * (1 - resp_filled)).T @ log_1mp  # (students, quad)

        # Posterior: P(theta_k | responses_j) ∝ P(responses_j | theta_k) * prior(theta_k)
        log_post = log_lik + log_weights[None, :]
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post)
        post /= post.sum(axis=1, keepdims=True)  # (students, quad)

        # ── M-step: fit each item's (a_i, b_i) ──
        for i in range(n_items):
            valid = valid_masks[i]
            if valid.sum() < 5:
                continue

            resp_i = resp_filled[i, valid]
            post_i = post[valid]  # (n_valid, quad)

            # Expected sufficient statistics at each quadrature point
            r_expected = post_i.T @ resp_i      # (quad,) — expected #correct
            n_expected = post_i.sum(axis=0)      # (quad,) — expected #responses

            def neg_mll(params):
                ai, bi = params
                if ai < 0.1:
                    return 1e8
                pk = expit(ai * (theta - bi))
                pk = np.clip(pk, 1e-10, 1 - 1e-10)
                ll = np.sum(r_expected * np.log(pk) + (n_expected - r_expected) * np.log(1 - pk))
                return -ll

            res = minimize(neg_mll, [a[i], b[i]], method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 0.005, 'fatol': 0.005})
            a[i] = np.clip(res.x[0], 0.2, 3.5)
            b[i] = np.clip(res.x[1], -5, 5)

        da = np.max(np.abs(a - a_old))
        db = np.max(np.abs(b - b_old))
        if verbose and (iteration % 5 == 0 or (da < tol and db < tol)):
            print(f'    EM iter {iteration+1}: max Δa={da:.4f}, max Δb={db:.4f}')
        if da < tol and db < tol:
            if verbose:
                print(f'    Converged at iteration {iteration+1}')
            break

    return a, b


def fit_2pl_irt(df, train_user_ids):
    """
    Fit 2PL IRT parameters using tiered approach.

    Returns DataFrame with columns:
      question_id, difficulty_2pl, discrimination_2pl, irt_method, n_responses,
      difficulty_logit (legacy), discrimination (legacy)
    """
    df_train = df[df['user_id'].isin(train_user_ids)].copy()

    # ── Response counts per question ──
    q_counts = df_train.groupby('question_id').size().reset_index(name='n_responses')
    all_questions = df_train['question_id'].unique()
    print(f'\nTotal questions in train set: {len(all_questions)}')

    # ── Tier assignment ──
    tier1_qids = set(q_counts[q_counts['n_responses'] >= MIN_RESPONSES_2PL]['question_id'])
    tier2_qids = set(q_counts[(q_counts['n_responses'] >= MIN_RESPONSES_LORD) &
                               (q_counts['n_responses'] < MIN_RESPONSES_2PL)]['question_id'])
    tier3_qids = set(q_counts[q_counts['n_responses'] < MIN_RESPONSES_LORD]['question_id'])
    # Include questions appearing only in test set
    all_q_in_data = set(df['question_id'].unique())
    tier3_qids = tier3_qids | (all_q_in_data - set(q_counts['question_id']))

    print(f'Tier 1 (≥{MIN_RESPONSES_2PL} responses, EM-MML 2PL): {len(tier1_qids)}')
    print(f'Tier 2 ({MIN_RESPONSES_LORD}-{MIN_RESPONSES_2PL-1} responses, Lord\'s approx): {len(tier2_qids)}')
    print(f'Tier 3 (<{MIN_RESPONSES_LORD} responses, prior): {len(tier3_qids)}')

    results = []

    # ══════════════════════════════════════════════════════════════════════
    # TIER 1: Proper 2PL via EM-MML (Bock & Aitkin 1981)
    # ══════════════════════════════════════════════════════════════════════
    if tier1_qids:
        df_t1 = df_train[df_train['question_id'].isin(tier1_qids)]
        # Take first attempt per student-question pair
        df_first = df_t1.sort_values('created_at').groupby(
            ['user_id', 'question_id'])['is_correct'].first().reset_index()
        df_first['is_correct'] = df_first['is_correct'].astype(float)
        pivot = df_first.pivot(index='question_id', columns='user_id', values='is_correct')
        resp_matrix = pivot.values.astype(np.float64)
        t1_qids_ordered = pivot.index.tolist()

        density = np.sum(~np.isnan(resp_matrix)) / resp_matrix.size
        print(f'\nFitting EM-MML 2PL on {resp_matrix.shape[0]} items × {resp_matrix.shape[1]} students '
              f'(density={density:.3f})...')

        disc_2pl, diff_2pl = _fit_2pl_em_mml(resp_matrix, n_quad=21, max_iter=50)

        # Build n_responses lookup for speed
        nresp_lookup = q_counts.set_index('question_id')['n_responses'].to_dict()

        for i, qid in enumerate(t1_qids_ordered):
            results.append({
                'question_id': qid,
                'difficulty_2pl': float(diff_2pl[i]),
                'discrimination_2pl': float(disc_2pl[i]),
                'irt_method': 'em_mml_2pl',
                'n_responses': int(nresp_lookup.get(qid, 0))
            })
        print(f'  EM-MML 2PL: a range [{disc_2pl.min():.3f}, {disc_2pl.max():.3f}], '
              f'b range [{diff_2pl.min():.3f}, {diff_2pl.max():.3f}]')

    # ══════════════════════════════════════════════════════════════════════
    # TIER 2: Lord's approximation from point-biserial correlation
    # ══════════════════════════════════════════════════════════════════════
    if tier2_qids:
        user_ability = df_train.groupby('user_id')['is_correct'].mean().rename('user_ability')
        df_with_ability = df_train.merge(user_ability, on='user_id')
        eps = 1e-6

        n_lord = 0
        for qid in tier2_qids:
            grp = df_with_ability[df_with_ability['question_id'] == qid]
            n_resp = len(grp)

            if n_resp >= MIN_RESPONSES_LORD and grp['is_correct'].std() > 0 and grp['user_ability'].std() > 0:
                r_pb, _ = sp_stats.pointbiserialr(grp['is_correct'].astype(int), grp['user_ability'])
            else:
                r_pb = 0.0

            # Lord's approximation: a ≈ r_pb / sqrt(1 - r_pb²)
            r_pb_clipped = np.clip(r_pb, -0.95, 0.95)
            if r_pb_clipped > 0:
                a_lord = r_pb_clipped / np.sqrt(1 - r_pb_clipped**2)
            else:
                a_lord = 0.5

            accuracy = grp['is_correct'].mean()
            accuracy_clipped = np.clip(accuracy, eps, 1 - eps)
            b_lord = np.clip(-np.log(accuracy_clipped / (1 - accuracy_clipped)), -5, 5)

            results.append({
                'question_id': qid,
                'difficulty_2pl': float(b_lord),
                'discrimination_2pl': float(np.clip(a_lord, 0.2, 3.0)),
                'irt_method': 'lord_approx',
                'n_responses': int(n_resp)
            })
            n_lord += 1

        del df_with_ability
        print(f'Lord\'s approximation: {n_lord} questions')

    # ══════════════════════════════════════════════════════════════════════
    # TIER 3: Population prior (insufficient data)
    # ══════════════════════════════════════════════════════════════════════
    nresp_lookup = q_counts.set_index('question_id')['n_responses'].to_dict()
    for qid in tier3_qids:
        results.append({
            'question_id': qid,
            'difficulty_2pl': 0.0,
            'discrimination_2pl': 1.0,
            'irt_method': 'prior',
            'n_responses': int(nresp_lookup.get(qid, 0))
        })
    print(f'Prior defaults: {len(tier3_qids)} questions')

    irt_df = pd.DataFrame(results)

    # ── Add legacy columns for backward compatibility ──
    eps = 1e-6
    q_legacy = (df_train.groupby('question_id')
                .agg(accuracy=('is_correct', 'mean'),
                     n=('is_correct', 'size'))
                .reset_index())
    q_legacy = q_legacy[q_legacy['n'] >= 5].copy()
    q_legacy['accuracy_clipped'] = q_legacy['accuracy'].clip(eps, 1 - eps)
    q_legacy['difficulty_logit'] = (-np.log(
        q_legacy['accuracy_clipped'] / (1 - q_legacy['accuracy_clipped']))).clip(-5, 5)

    user_ability = df_train.groupby('user_id')['is_correct'].mean().rename('user_ability')
    df_wa = df_train.merge(user_ability, on='user_id')
    disc_records = []
    for qid, grp in df_wa.groupby('question_id'):
        if len(grp) < 5 or grp['is_correct'].std() == 0 or grp['user_ability'].std() == 0:
            disc_records.append({'question_id': qid, 'discrimination': 0.0})
            continue
        r, _ = sp_stats.pointbiserialr(grp['is_correct'].astype(int), grp['user_ability'])
        disc_records.append({'question_id': qid, 'discrimination': max(r, 0.0)})
    legacy_disc = pd.DataFrame(disc_records)
    del df_wa

    q_legacy = q_legacy[['question_id', 'difficulty_logit']].merge(legacy_disc, on='question_id', how='left')
    q_legacy['discrimination'] = q_legacy['discrimination'].fillna(0)

    irt_df = irt_df.merge(q_legacy[['question_id', 'difficulty_logit', 'discrimination']],
                          on='question_id', how='left')
    irt_df['difficulty_logit'] = irt_df['difficulty_logit'].fillna(0.0)
    irt_df['discrimination'] = irt_df['discrimination'].fillna(0.0)

    return irt_df


def compare_irt_methods(irt_df):
    """Compare 2PL vs legacy IRT parameters — rank correlations."""
    valid = irt_df[(irt_df['irt_method'] != 'prior') & (irt_df['discrimination'] > 0)].copy()
    print(f'\n=== IRT Method Comparison (n={len(valid)}) ===')

    r_diff_spear, _ = sp_stats.spearmanr(valid['difficulty_2pl'], valid['difficulty_logit'])
    r_diff_pears, _ = sp_stats.pearsonr(valid['difficulty_2pl'], valid['difficulty_logit'])
    print(f'Difficulty:      Spearman ρ={r_diff_spear:.4f}, Pearson r={r_diff_pears:.4f}')

    r_disc_spear, _ = sp_stats.spearmanr(valid['discrimination_2pl'], valid['discrimination'])
    r_disc_pears, _ = sp_stats.pearsonr(valid['discrimination_2pl'], valid['discrimination'])
    print(f'Discrimination:  Spearman ρ={r_disc_spear:.4f}, Pearson r={r_disc_pears:.4f}')

    for method in ['em_mml_2pl', 'lord_approx']:
        subset = valid[valid['irt_method'] == method]
        if len(subset) > 5:
            rs, _ = sp_stats.spearmanr(subset['discrimination_2pl'], subset['discrimination'])
            rd, _ = sp_stats.spearmanr(subset['difficulty_2pl'], subset['difficulty_logit'])
            print(f'  {method} (n={len(subset)}): disc ρ={rs:.4f}, diff ρ={rd:.4f}')

    return {
        'difficulty_spearman': r_diff_spear,
        'difficulty_pearson': r_diff_pears,
        'discrimination_spearman': r_disc_spear,
        'discrimination_pearson': r_disc_pears,
    }


# ═══════════════════════════════════════════════════════════════════════
# EAP (Expected A Posteriori) ability estimation
# ═══════════════════════════════════════════════════════════════════════

def _icc_2pl(theta, a, b):
    """2PL item characteristic curve: P(correct | theta, a, b)."""
    return expit(a * (theta - b))


def compute_eap_theta(responses, a_params, b_params, n_quad=41):
    """
    Compute EAP theta from a sequence of responses.

    Returns: (theta_eap, se_theta)
    """
    theta_grid = np.linspace(-4, 4, n_quad)
    log_prior = -0.5 * theta_grid**2 - 0.5 * np.log(2 * np.pi)

    log_lik = np.zeros(n_quad)
    for resp, a, b in zip(responses, a_params, b_params):
        p = _icc_2pl(theta_grid, a, b)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        if resp == 1:
            log_lik += np.log(p)
        else:
            log_lik += np.log(1 - p)

    log_post = log_prior + log_lik
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= np.sum(post)

    theta_eap = np.sum(theta_grid * post)
    se_theta = np.sqrt(np.sum((theta_grid - theta_eap)**2 * post))

    return theta_eap, se_theta


def compute_running_eap(df, q_irt, n_quad=41):
    """
    Compute running EAP theta for each interaction (strictly causal).

    For interaction at position t, EAP is computed from responses 0..t-1.
    First interaction gets theta=0 (prior mean).
    Uses efficient incremental log-posterior update (O(n_quad) per interaction).
    """
    theta_grid = np.linspace(-4, 4, n_quad)
    log_prior = -0.5 * theta_grid**2 - 0.5 * np.log(2 * np.pi)

    q_a = q_irt.set_index('question_id')['discrimination_2pl'].to_dict()
    q_b = q_irt.set_index('question_id')['difficulty_2pl'].to_dict()
    default_a, default_b = 1.0, 0.0

    eap_values = np.zeros(len(df))
    current_user = None
    log_post = None

    for i, (idx, row) in enumerate(df.iterrows()):
        uid = row['user_id']
        qid = row['question_id']
        correct = int(row['is_correct'])

        if uid != current_user:
            current_user = uid
            log_post = log_prior.copy()
            eap_values[i] = 0.0
        else:
            post_shifted = log_post - np.max(log_post)
            post_norm = np.exp(post_shifted)
            post_norm /= np.sum(post_norm)
            eap_values[i] = np.sum(theta_grid * post_norm)

        a = q_a.get(qid, default_a)
        b = q_b.get(qid, default_b)
        p = _icc_2pl(theta_grid, a, b)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        if correct:
            log_post += np.log(p)
        else:
            log_post += np.log(1 - p)

        if i > 0 and i % 20000 == 0:
            print(f'  EAP progress: {i:,}/{len(df):,} interactions')

    return pd.Series(eap_values, index=df.index, name='student_ability_eap')


# ═══════════════════════════════════════════════════════════════════════
# MAIN — run as script to fit IRT and save
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time

    print('=' * 70)
    print('IRT FITTING: 2PL via EM-MML + Lord\'s approximation')
    print('=' * 70)

    df = pd.read_csv('data/students_969.csv', parse_dates=['created_at', 'submitted_at'])
    print(f'Loaded: {len(df):,} interactions, {df.user_id.nunique()} students')

    train_ids, test_ids = get_train_test_split(df)
    print(f'Train: {len(train_ids)}, Test: {len(test_ids)}')

    t0 = time.time()
    irt_2pl = fit_2pl_irt(df, train_ids)
    t1 = time.time()
    print(f'\nFitting time: {t1-t0:.1f}s')

    irt_2pl.to_csv('data/question_irt_2pl.csv', index=False)
    print(f'Saved data/question_irt_2pl.csv: {len(irt_2pl)} questions')

    print(f'\n=== 2PL Parameter Summary ===')
    for method in ['em_mml_2pl', 'lord_approx', 'prior']:
        subset = irt_2pl[irt_2pl['irt_method'] == method]
        if len(subset) > 0:
            print(f'{method} (n={len(subset)}):')
            print(f'  difficulty_2pl:      mean={subset["difficulty_2pl"].mean():.3f}, '
                  f'std={subset["difficulty_2pl"].std():.3f}, '
                  f'range=[{subset["difficulty_2pl"].min():.3f}, {subset["difficulty_2pl"].max():.3f}]')
            print(f'  discrimination_2pl:  mean={subset["discrimination_2pl"].mean():.3f}, '
                  f'std={subset["discrimination_2pl"].std():.3f}, '
                  f'range=[{subset["discrimination_2pl"].min():.3f}, {subset["discrimination_2pl"].max():.3f}]')

    correlations = compare_irt_methods(irt_2pl)

    print(f'\nDone. Output: data/question_irt_2pl.csv')
