#!/usr/bin/env python3
"""Comprehensive test: 20 most active students — all metrics validated.

Tests:
1. Mastery convergence and range checks per subject
2. Confidence values — cold start, convergence, time-gap decay
3. KT calibration — predicted P(correct) vs actual outcomes (Brier, AUC)
4. Velocity trackers — sign agreement, range checks, correlations
5. FSRS memory dynamics — stability growth, retrievability decay
6. Cold start behavior — first 5 interactions vs actual outcomes
7. MVS aggregate scores — range, monotonicity with improvement
8. Mastery vs actual accuracy per subject
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import math
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# ── Load data (replicate app_velocity_demo.py load_data without Streamlit) ───

print("Loading data...")
df = pd.read_csv('data/students_969.csv', parse_dates=['created_at', 'submitted_at'])

train_user_ids = set(pd.read_csv('data/train_user_ids.csv')['user_id'])
test_user_ids = set(pd.read_csv('data/test_user_ids.csv')['user_id'])
q_irt_legacy = pd.read_csv('data/question_irt_train.csv')
q_irt_2pl = pd.read_csv('data/question_irt_2pl.csv')

# Merge IRT
for c in ['difficulty_logit', 'discrimination']:
    if c in df.columns:
        df = df.drop(columns=[c])
df = df.merge(q_irt_legacy, on='question_id', how='left')
df['difficulty_logit'] = df['difficulty_logit'].fillna(0.0).clip(-5, 5)
df['discrimination'] = df['discrimination'].fillna(0).clip(lower=0)

irt_2pl_cols = q_irt_2pl[['question_id', 'discrimination_2pl', 'difficulty_2pl']].copy()
df = df.merge(irt_2pl_cols, on='question_id', how='left')
df['discrimination_2pl'] = df['discrimination_2pl'].fillna(1.0).clip(0.1, 5.0)
df['difficulty_2pl'] = df['difficulty_2pl'].fillna(0.0).clip(-5, 5)

# EAP student ability
from irt_fitting import compute_running_eap
df = df.sort_values(['user_id', 'created_at']).reset_index(drop=True)
print("Computing EAP student abilities...")
df['student_ability_logit'] = compute_running_eap(df, q_irt_2pl, n_quad=41)

# Encoders
for col, new_col in [('user_id', 'user_idx'), ('subject', 'subject_idx')]:
    enc = LabelEncoder()
    df[new_col] = enc.fit_transform(df[col].fillna('Unknown'))

df['question_pattern_clean'] = df['question_pattern'].fillna('Unknown')
pattern_encoder = LabelEncoder()
df['pattern_idx'] = pattern_encoder.fit_transform(df['question_pattern_clean'])
n_patterns = int(df['pattern_idx'].nunique())

df['elapsed_log'] = np.log1p(df['elapsed_seconds'].fillna(0).clip(0, 86400*30))
df['subject_elapsed_log'] = np.log1p(df['subject_elapsed_seconds'].fillna(0).clip(0, 86400*30))
df['topic_elapsed_log'] = np.log1p(df['topic_elapsed_seconds'].fillna(0).clip(0, 86400*30))
df['subject_accuracy_prior'] = df['subject_accuracy_prior'].fillna(0.5)
df['topic_accuracy_prior'] = df['topic_accuracy_prior'].fillna(0.5)
df['time_spent_log'] = np.log1p(df['total_time_spent'].fillna(0).clip(0, 3600))
df['subject_attempts_log'] = np.log1p(df['subject_attempts_prior'].fillna(0))
df['topic_attempts_log'] = np.log1p(df['topic_attempts_prior'].fillna(0))
df['correct'] = df['is_correct'].astype(int)

feature_cols = [
    'elapsed_log', 'subject_elapsed_log', 'topic_elapsed_log',
    'subject_accuracy_prior', 'topic_accuracy_prior',
    'time_spent_log', 'subject_attempts_log', 'topic_attempts_log',
    'difficulty_logit', 'discrimination',
    'student_ability_logit',
]

train_mask = df['user_id'].isin(train_user_ids)
cont_mean = df.loc[train_mask, feature_cols].mean()
cont_std = df.loc[train_mask, feature_cols].std().replace(0, 1)

n_skills = int(df['subject_idx'].max() + 1)
subjects = sorted(df['subject'].unique())
taxonomy = {s: s for s in subjects}

# ── Load SAINT-Lite model ───
print("Loading SAINT-Lite model...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from app_velocity_demo import SAINTLite, get_student_predictions

model = SAINTLite(n_skills, n_patterns, len(feature_cols),
                  embed_dim=56, n_heads=4, n_layers=2, ff_dim=128, dropout=0.2)
dummy = {
    'interactions': tf.zeros((1, 10), dtype='int32'),
    'patterns': tf.zeros((1, 10), dtype='int32'),
    'forget_features': tf.zeros((1, 10, len(feature_cols)), dtype='float32'),
}
model(dummy, training=False)
model.load_weights('models/saint_subject_969_v3_fold0.weights.h5')
print("Model loaded.")

# ── Select top 20 most active students ───
from mastery_velocity import MultiVelocityPipeline

top20_counts = df.groupby('user_id').size().nlargest(20)
top20_uids = list(top20_counts.index)
uid_to_idx = dict(zip(df['user_id'], df['user_idx']))

print(f"\nProcessing {len(top20_uids)} most active students ({top20_counts.sum()} total interactions)...\n")

# ══════════════════════════════════════════════════════════════════════
# Process each student and collect all metrics
# ══════════════════════════════════════════════════════════════════════

all_results = {}  # uid -> result dict

for rank, uid in enumerate(top20_uids):
    user_idx = uid_to_idx[uid]
    n_interactions = top20_counts[uid]

    # Get KT predictions
    interactions_df, predictions = get_student_predictions(
        model, df, feature_cols, cont_mean, cont_std, user_idx)
    if interactions_df is None:
        print(f"  [{rank+1}] {uid[:8]}... SKIPPED (< 2 interactions)")
        continue

    # Process through the full pipeline
    pipeline = MultiVelocityPipeline(taxonomy)
    real_uid = str(uid)
    ts0 = interactions_df['submitted_at'].iloc[0].timestamp() / 86400.0

    # Per-interaction tracking
    timeline = []
    per_subject_first5 = defaultdict(list)  # subject -> list of (is_correct, kt_pred, mastery)
    per_subject_all = defaultdict(list)

    for i in range(len(interactions_df)):
        row = interactions_df.iloc[i]
        topic = row['subject']
        is_correct = bool(row['correct'])
        ts_days = row['submitted_at'].timestamp() / 86400.0 - ts0
        kt_pred = float(predictions[i])
        disc_2pl = float(row.get('discrimination_2pl', 1.0))
        diff_2pl = float(row.get('difficulty_2pl', 0.0))

        result = pipeline.process_interaction(
            student_id=real_uid,
            topic_id=topic,
            is_correct=is_correct,
            timestamp_days=ts_days,
            kt_prediction=kt_pred,
            discrimination=disc_2pl,
            difficulty=diff_2pl,
        )

        # Get Kalman state for this topic
        ts_state = pipeline.base_pipeline.mastery._peek_topic_state(real_uid, topic)
        confidence = ts_state.confidence if ts_state else 0.0
        fsrs_stab = ts_state.fsrs_state.stability if ts_state else 0.0
        n_topic = ts_state.n_interactions if ts_state else 0

        timeline.append({
            'i': i,
            'subject': topic,
            'is_correct': is_correct,
            'kt_prediction': kt_pred,
            'topic_mastery': result['topic_mastery'],
            'overall_mastery': result['overall_mastery'],
            'confidence': confidence,
            'fsrs_stability': fsrs_stab,
            'ts_days': ts_days,
            'v_baseline': result['velocities']['baseline'],
            'v_zpdes': result['velocities']['zpdes'],
            'v_kt': result['velocities']['kt'],
            'v_cusum': result['velocities']['cusum'],
            'v_mastery_delta': result['velocities']['mastery_delta'],
            'v_regression': result['velocities']['regression'],
            'v_ensemble': result['velocities']['ensemble'],
            'n_topic_interactions': n_topic,
        })

        per_subject_all[topic].append({
            'is_correct': is_correct,
            'kt_pred': kt_pred,
            'mastery': result['topic_mastery'],
            'confidence': confidence,
        })
        if n_topic <= 5:
            per_subject_first5[topic].append({
                'is_correct': is_correct,
                'kt_pred': kt_pred,
                'mastery': result['topic_mastery'],
                'confidence': confidence,
            })

    # Get MVS scores
    current_ts = interactions_df['submitted_at'].iloc[-1].timestamp() / 86400.0 - ts0
    mvs_all = pipeline.get_mvs_all(real_uid, current_ts)

    all_results[uid] = {
        'n_interactions': len(interactions_df),
        'timeline': timeline,
        'per_subject_first5': dict(per_subject_first5),
        'per_subject_all': dict(per_subject_all),
        'mvs_all': mvs_all,
        'pipeline': pipeline,
        'real_uid': real_uid,
        'current_ts': current_ts,
        'actual_acc': interactions_df['correct'].mean(),
    }
    print(f"  [{rank+1:>2}] {uid[:8]}... {len(interactions_df):>4} interactions, "
          f"acc={interactions_df['correct'].mean():.1%}, "
          f"mastery={timeline[-1]['overall_mastery']:.3f}, "
          f"confidence={timeline[-1]['confidence']:.3f}")

print(f"\nProcessed {len(all_results)} students.")

# ══════════════════════════════════════════════════════════════════════
# TEST 1: Mastery Range & Convergence
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 1: MASTERY RANGE & CONVERGENCE")
print("="*70)

test1_pass = True
for uid, res in all_results.items():
    tl = res['timeline']
    final_mastery = tl[-1]['overall_mastery']
    final_topic_m = tl[-1]['topic_mastery']

    # Range check: mastery in [0, 1]
    for t in tl:
        if not (0.0 <= t['topic_mastery'] <= 1.0):
            print(f"  FAIL: {uid[:8]} topic_mastery={t['topic_mastery']:.4f} out of [0,1]")
            test1_pass = False
        if not (0.0 <= t['overall_mastery'] <= 1.0):
            print(f"  FAIL: {uid[:8]} overall_mastery={t['overall_mastery']:.4f} out of [0,1]")
            test1_pass = False

    # Convergence: overall mastery should be > 0.3 after 100+ interactions for non-random students
    if len(tl) > 100 and res['actual_acc'] > 0.5 and final_mastery < 0.3:
        print(f"  WARN: {uid[:8]} final_mastery={final_mastery:.3f} low despite {len(tl)} interactions, acc={res['actual_acc']:.1%}")

if test1_pass:
    print("  All mastery values in [0, 1] range.")

# Direction check: high-accuracy students should have high mastery
accs = [(uid, res['actual_acc'], res['timeline'][-1]['overall_mastery']) for uid, res in all_results.items()]
accs.sort(key=lambda x: x[1])
print(f"\n  Accuracy vs Final Mastery (rank correlation):")
from scipy import stats
acc_vals = [a[1] for a in accs]
mas_vals = [a[2] for a in accs]
r_mastery_acc, p_mastery_acc = stats.spearmanr(acc_vals, mas_vals)
print(f"  Spearman r = {r_mastery_acc:.3f}, p = {p_mastery_acc:.4f}")
print(f"  {'PASS' if r_mastery_acc > 0.3 else 'FAIL'}: Mastery correlates {'well' if r_mastery_acc > 0.5 else 'weakly' if r_mastery_acc > 0.3 else 'poorly'} with accuracy")

# ══════════════════════════════════════════════════════════════════════
# TEST 2: CONFIDENCE CHECKS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: CONFIDENCE VALUES")
print("="*70)

test2_pass = True
cold_start_confs = []
final_confs = []

for uid, res in all_results.items():
    tl = res['timeline']

    # Range check
    for t in tl:
        if not (0.0 <= t['confidence'] <= 1.0):
            print(f"  FAIL: {uid[:8]} confidence={t['confidence']:.4f} out of [0,1]")
            test2_pass = False

    # Cold start: first interaction confidence should be low (<= 0.55)
    first_conf = tl[0]['confidence']
    cold_start_confs.append(first_conf)
    if first_conf > 0.6:
        print(f"  WARN: {uid[:8]} cold-start confidence={first_conf:.3f} (high for first interaction)")

    # Final confidence after many interactions
    final_confs.append(tl[-1]['confidence'])

if test2_pass:
    print("  All confidence values in [0, 1] range.")

print(f"\n  Cold-start confidence: mean={np.mean(cold_start_confs):.3f}, "
      f"min={np.min(cold_start_confs):.3f}, max={np.max(cold_start_confs):.3f}")
print(f"  Final confidence:     mean={np.mean(final_confs):.3f}, "
      f"min={np.min(final_confs):.3f}, max={np.max(final_confs):.3f}")
print(f"  {'PASS' if np.mean(final_confs) > np.mean(cold_start_confs) else 'FAIL'}: "
      f"Confidence grows with interactions (cold={np.mean(cold_start_confs):.3f} -> final={np.mean(final_confs):.3f})")

# Confidence time-gap decay check (using stored pipeline states)
print("\n  Time-gap confidence decay (sample):")
n_gap_tests = 0
n_gap_pass = 0
for uid, res in list(all_results.items())[:5]:
    pip = res['pipeline']
    real_uid = res['real_uid']
    curr_ts = res['current_ts']
    # Check one subject per student
    for subj in list(res['per_subject_all'].keys())[:1]:
        conf_now = pip.base_pipeline.mastery.get_topic_confidence(real_uid, subj, curr_ts)
        conf_later = pip.base_pipeline.mastery.get_topic_confidence(real_uid, subj, curr_ts + 30)
        n_gap_tests += 1
        if conf_later <= conf_now + 1e-10:
            n_gap_pass += 1
        else:
            print(f"    FAIL: {uid[:8]} {subj}: conf@now={conf_now:.4f} -> conf@+30d={conf_later:.4f} (increased!)")
print(f"  Gap decay: {n_gap_pass}/{n_gap_tests} passed")

# ══════════════════════════════════════════════════════════════════════
# TEST 3: KT PREDICTION CALIBRATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 3: KT PREDICTION CALIBRATION")
print("="*70)

all_kt_preds = []
all_actuals = []
per_student_brier = []

for uid, res in all_results.items():
    tl = res['timeline']
    preds_s = [t['kt_prediction'] for t in tl]
    acts_s = [float(t['is_correct']) for t in tl]
    all_kt_preds.extend(preds_s)
    all_actuals.extend(acts_s)
    brier = np.mean([(p - a)**2 for p, a in zip(preds_s, acts_s)])
    per_student_brier.append(brier)

all_kt_preds = np.array(all_kt_preds)
all_actuals = np.array(all_actuals)

# Overall Brier score
brier_overall = np.mean((all_kt_preds - all_actuals)**2)
print(f"  Overall Brier score: {brier_overall:.4f} (lower is better, <0.25 = better than coin flip)")
print(f"  {'PASS' if brier_overall < 0.25 else 'FAIL'}: KT predictions better than random")

# AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(all_actuals, all_kt_preds)
print(f"  Overall AUC: {auc:.4f}")
print(f"  {'PASS' if auc > 0.6 else 'FAIL'}: AUC > 0.6")

# Calibration by decile
print(f"\n  Calibration by prediction decile:")
deciles = np.percentile(all_kt_preds, np.arange(0, 100, 10))
for i in range(10):
    lo = deciles[i]
    hi = deciles[i+1] if i < 9 else 1.01
    mask = (all_kt_preds >= lo) & (all_kt_preds < hi)
    if mask.sum() > 0:
        pred_mean = all_kt_preds[mask].mean()
        act_mean = all_actuals[mask].mean()
        n_bin = mask.sum()
        cal_err = abs(pred_mean - act_mean)
        print(f"    Decile {i+1}: pred={pred_mean:.3f}, actual={act_mean:.3f}, "
              f"|err|={cal_err:.3f}, n={n_bin} {'!' if cal_err > 0.15 else ''}")

# Per-student Brier
print(f"\n  Per-student Brier: mean={np.mean(per_student_brier):.4f}, "
      f"min={np.min(per_student_brier):.4f}, max={np.max(per_student_brier):.4f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 4: COLD START — First 5 interactions per subject
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 4: COLD START BEHAVIOR")
print("="*70)

cold_kt_preds = []
cold_actuals = []
cold_masteries = []
cold_confs = []
n_cold_subjects = 0

for uid, res in all_results.items():
    for subj, entries in res['per_subject_first5'].items():
        n_cold_subjects += 1
        for entry in entries:
            cold_kt_preds.append(entry['kt_pred'])
            cold_actuals.append(float(entry['is_correct']))
            cold_masteries.append(entry['mastery'])
            cold_confs.append(entry['confidence'])

cold_kt_preds = np.array(cold_kt_preds)
cold_actuals = np.array(cold_actuals)
cold_masteries = np.array(cold_masteries)
cold_confs = np.array(cold_confs)

print(f"  Cold start data: {len(cold_actuals)} interactions across {n_cold_subjects} subject-student pairs")

# KT calibration during cold start
cold_brier = np.mean((cold_kt_preds - cold_actuals)**2)
cold_auc = roc_auc_score(cold_actuals, cold_kt_preds) if len(np.unique(cold_actuals)) > 1 else 0.5
print(f"  Cold-start KT Brier: {cold_brier:.4f}")
print(f"  Cold-start KT AUC:   {cold_auc:.4f}")

# Mastery should start near 0.5 (prior)
print(f"\n  Cold-start mastery: mean={cold_masteries.mean():.3f}, "
      f"min={cold_masteries.min():.3f}, max={cold_masteries.max():.3f}")
print(f"  {'PASS' if 0.3 < cold_masteries.mean() < 0.7 else 'WARN'}: "
      f"Cold-start mastery centered near prior")

# Confidence during cold start should be low
print(f"  Cold-start confidence: mean={cold_confs.mean():.3f}, "
      f"min={cold_confs.min():.3f}, max={cold_confs.max():.3f}")
print(f"  {'PASS' if cold_confs.mean() < 0.6 else 'WARN'}: "
      f"Cold-start confidence appropriately low")

# ══════════════════════════════════════════════════════════════════════
# TEST 5: VELOCITY TRACKERS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 5: VELOCITY TRACKERS")
print("="*70)

# Collect final velocities per tracker per student
tracker_names = ['v_baseline', 'v_zpdes', 'v_kt', 'v_cusum', 'v_mastery_delta', 'v_ensemble']
velocity_matrix = {name: [] for name in tracker_names}

for uid, res in all_results.items():
    tl = res['timeline']
    last = tl[-1]
    for name in tracker_names:
        velocity_matrix[name].append(last[name])

# Range check: all velocities should be finite
print("  Range checks:")
all_ok = True
for name in tracker_names:
    vals = velocity_matrix[name]
    v_min, v_max = min(vals), max(vals)
    has_nan = any(math.isnan(v) or math.isinf(v) for v in vals)
    status = "FAIL" if has_nan else "OK"
    if has_nan:
        all_ok = False
    print(f"    {name:>18}: min={v_min:+.4f}, max={v_max:+.4f}, finite={status}")
print(f"  {'PASS' if all_ok else 'FAIL'}: All velocities finite")

# Correlation matrix between trackers
print(f"\n  Velocity correlation matrix (Spearman):")
corr_header = "             " + "  ".join(f"{n[2:]:>8}" for n in tracker_names)
print(f"    {corr_header}")
for i, name_i in enumerate(tracker_names):
    row = f"    {name_i[2:]:>10}  "
    for j, name_j in enumerate(tracker_names):
        if j <= i:
            r, _ = stats.spearmanr(velocity_matrix[name_i], velocity_matrix[name_j])
            row += f"  {r:>+.3f}  "
        else:
            row += "         "
    print(row)

# Sign agreement: for improving students (acc > 60%), velocities should be positive on average
improving = [uid for uid, res in all_results.items() if res['actual_acc'] > 0.6]
if improving:
    print(f"\n  Sign check for {len(improving)} improving students (acc > 60%):")
    for name in tracker_names:
        vals = [velocity_matrix[name][list(all_results.keys()).index(uid)] for uid in improving]
        mean_v = np.mean(vals)
        pct_positive = sum(1 for v in vals if v > 0) / len(vals)
        print(f"    {name:>18}: mean={mean_v:+.4f}, {pct_positive:.0%} positive")

# ══════════════════════════════════════════════════════════════════════
# TEST 6: FSRS MEMORY DYNAMICS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 6: FSRS MEMORY DYNAMICS")
print("="*70)

stab_correct = []   # stability after correct answer
stab_incorrect = []  # stability after incorrect answer

for uid, res in all_results.items():
    tl = res['timeline']
    for i in range(1, len(tl)):
        if tl[i]['subject'] == tl[i-1]['subject']:
            if tl[i]['is_correct']:
                stab_correct.append(tl[i]['fsrs_stability'])
            else:
                stab_incorrect.append(tl[i]['fsrs_stability'])

print(f"  Stability after correct:   mean={np.mean(stab_correct):.2f}, "
      f"median={np.median(stab_correct):.2f}")
print(f"  Stability after incorrect: mean={np.mean(stab_incorrect):.2f}, "
      f"median={np.median(stab_incorrect):.2f}")
print(f"  {'PASS' if np.mean(stab_correct) > np.mean(stab_incorrect) else 'FAIL'}: "
      f"Correct answers lead to higher stability")

# Check stability growth per subject
print(f"\n  Sample stability trajectories:")
for uid in list(all_results.keys())[:3]:
    tl = all_results[uid]['timeline']
    subjs = list(all_results[uid]['per_subject_all'].keys())[:1]
    for subj in subjs:
        subj_tl = [t for t in tl if t['subject'] == subj]
        if len(subj_tl) >= 5:
            stabs = [t['fsrs_stability'] for t in subj_tl]
            print(f"    {uid[:8]} / {subj[:15]:>15}: "
                  f"stab=[{stabs[0]:.1f}, {stabs[len(stabs)//4]:.1f}, "
                  f"{stabs[len(stabs)//2]:.1f}, {stabs[-1]:.1f}] "
                  f"(first->last, n={len(subj_tl)})")

# ══════════════════════════════════════════════════════════════════════
# TEST 7: MVS AGGREGATE SCORES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 7: MVS AGGREGATE SCORES")
print("="*70)

approach_names = ['baseline', 'zpdes', 'kt', 'cusum', 'mastery_delta', 'ensemble']

print(f"\n  {'Student':>10}  {'Acc':>5}", end="")
for app in approach_names:
    print(f"  {app[:8]:>8}", end="")
print()
print(f"  {'--------':>10}  {'---':>5}", end="")
for _ in approach_names:
    print(f"  {'--------':>8}", end="")
print()

mvs_by_approach = {app: [] for app in approach_names}
accs_for_corr = []

for uid, res in all_results.items():
    mvs = res['mvs_all']
    acc = res['actual_acc']
    accs_for_corr.append(acc)
    short_id = uid[:8] + ".."
    print(f"  {short_id:>10}  {acc:>5.1%}", end="")
    for app in approach_names:
        m = mvs[app]['mvs']
        mvs_by_approach[app].append(m)
        print(f"  {m:>8.4f}", end="")
    print()

# MVS range check
print(f"\n  MVS range check:")
all_in_range = True
for app in approach_names:
    vals = mvs_by_approach[app]
    v_min, v_max = min(vals), max(vals)
    ok = all(0.0 <= v <= 1.0 for v in vals)
    if not ok:
        all_in_range = False
    print(f"    {app:>15}: [{v_min:.4f}, {v_max:.4f}] {'OK' if ok else 'OUT OF RANGE'}")
print(f"  {'PASS' if all_in_range else 'FAIL'}: All MVS in [0, 1]")

# MVS correlation with accuracy
print(f"\n  MVS correlation with actual accuracy:")
for app in approach_names:
    r, p = stats.spearmanr(accs_for_corr, mvs_by_approach[app])
    print(f"    {app:>15}: Spearman r={r:+.3f} (p={p:.4f})")

# ══════════════════════════════════════════════════════════════════════
# TEST 8: MASTERY vs ACTUAL ACCURACY PER SUBJECT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 8: MASTERY vs ACTUAL ACCURACY (per subject)")
print("="*70)

subj_mastery_vals = []
subj_acc_vals = []

for uid, res in all_results.items():
    for subj, entries in res['per_subject_all'].items():
        if len(entries) < 5:
            continue
        final_mastery = entries[-1]['mastery']
        actual_acc = np.mean([e['is_correct'] for e in entries])
        subj_mastery_vals.append(final_mastery)
        subj_acc_vals.append(actual_acc)

r_subj, p_subj = stats.spearmanr(subj_mastery_vals, subj_acc_vals)
print(f"  Subject-level (n={len(subj_mastery_vals)} student-subject pairs with >= 5 interactions)")
print(f"  Mastery vs Accuracy: Spearman r = {r_subj:.3f} (p = {p_subj:.6f})")
print(f"  {'PASS' if r_subj > 0.3 else 'FAIL'}: Mastery tracks actual accuracy (r={r_subj:.3f})")

# Mastery calibration: bin by mastery decile
print(f"\n  Mastery calibration (mastery vs actual accuracy):")
subj_mastery_arr = np.array(subj_mastery_vals)
subj_acc_arr = np.array(subj_acc_vals)
for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
    mask = (subj_mastery_arr >= lo) & (subj_mastery_arr < hi)
    if mask.sum() > 0:
        mean_m = subj_mastery_arr[mask].mean()
        mean_a = subj_acc_arr[mask].mean()
        err = abs(mean_m - mean_a)
        print(f"    Mastery [{lo:.2f},{hi:.2f}): pred={mean_m:.3f}, actual_acc={mean_a:.3f}, "
              f"|err|={err:.3f}, n={mask.sum()} {'!' if err > 0.2 else ''}")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
  Students tested:         {len(all_results)}
  Total interactions:      {sum(r['n_interactions'] for r in all_results.values())}

  KT Model:
    Overall Brier:         {brier_overall:.4f}
    Overall AUC:           {auc:.4f}
    Cold-start Brier:      {cold_brier:.4f}
    Cold-start AUC:        {cold_auc:.4f}

  Mastery:
    Mastery-Accuracy r:    {r_mastery_acc:.3f}
    Subject-level r:       {r_subj:.3f}
    Cold-start mastery:    {cold_masteries.mean():.3f} (should be ~0.5)

  Confidence:
    Cold-start mean:       {np.mean(cold_start_confs):.3f}
    Final mean:            {np.mean(final_confs):.3f}

  FSRS:
    Stability (correct):   {np.mean(stab_correct):.1f} days
    Stability (incorrect): {np.mean(stab_incorrect):.1f} days

  MVS range:               [{min(min(v) for v in mvs_by_approach.values()):.4f}, {max(max(v) for v in mvs_by_approach.values()):.4f}]
""")
print("Done.")
