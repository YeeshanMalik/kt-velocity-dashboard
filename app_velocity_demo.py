"""
Streamlit Demo: Knowledge Tracing + Velocity Dashboard

Sections:
  1. Student Overview — metrics + cumulative accuracy chart
  2. SAINT-Lite Model — actual vs predicted, calibration
  3. Mastery State (EKF) — per-subject theta, alpha, mastery, confidence
  4. Velocity Dashboard — KT + Mastery Delta ensemble (both difficulty-aware)
  5. Per-Subject Velocity Table — color-coded by sign
  6. Velocity Over Time — Plotly line chart
  7. Algorithm Explanations — Ensemble, Mastery Delta, KT Logit
  8. Recommendation Scoring (ITZS) — ELG/Review/InfoGain/Novelty
  9. Aggregate Metrics — optional cross-student comparison
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mastery_velocity import (
    MultiVelocityPipeline, normalize_velocity, compute_breadth, compute_mvs,
    score_topics_for_recommendation, zpd_score, review_urgency_sigmoid,
    information_gain, ZPD_PEAK, ZPD_SIGMA, REVIEW_R_THRESHOLD, REVIEW_SIGMOID_K,
    NOVELTY_DECAY_N, ITZS_WEIGHTS,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KT Velocity Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════
# SAINT-LITE MODEL (subject-level, on 969 students)
# ═══════════════════════════════════════════════════════════════════════

class SAINTLite(keras.Model):
    def __init__(self, n_skills, n_patterns, n_features,
                 embed_dim=56, n_heads=4, n_layers=2, ff_dim=128, dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_skills = n_skills
        self.embed_dim = embed_dim
        self._n_layers = n_layers

        self.inter_emb = layers.Embedding(n_skills * 2, embed_dim)
        self.pattern_emb = layers.Embedding(n_patterns, embed_dim // 4)
        self.feat_proj = layers.Dense(embed_dim // 4)
        self.input_proj = layers.Dense(embed_dim)
        self.time_proj = layers.Dense(embed_dim, activation='tanh')
        self.pos_proj = layers.Dense(embed_dim)
        self.input_dropout = layers.Dropout(dropout)

        self.attn_layers = []
        self.ffn_layers = []
        self.ln1_layers = []
        self.ln2_layers = []
        self.attn_dropouts = []
        # KerpleLog attention bias parameters
        self.kerple_log_p = []
        self.kerple_log_a = []

        for i in range(n_layers):
            self.attn_layers.append(
                layers.MultiHeadAttention(
                    num_heads=n_heads, key_dim=embed_dim // n_heads, dropout=dropout))
            self.ffn_layers.append(keras.Sequential([
                layers.Dense(ff_dim, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]))
            self.ln1_layers.append(layers.LayerNormalization(epsilon=1e-6))
            self.ln2_layers.append(layers.LayerNormalization(epsilon=1e-6))
            self.attn_dropouts.append(layers.Dropout(dropout))
            self.kerple_log_p.append(
                tf.Variable(
                    tf.constant(0.5, shape=(n_heads, 1, 1)),
                    trainable=True, name=f'kerple_log_p_{i}'))
            self.kerple_log_a.append(
                tf.Variable(
                    tf.constant(0.0, shape=(n_heads, 1, 1)),
                    trainable=True, name=f'kerple_log_a_{i}'))

        self.output_dense = layers.Dense(n_skills)

    def _build_attention_bias(self, seq_len, layer_idx):
        positions = tf.cast(tf.range(seq_len), tf.float32)
        dist = tf.maximum(positions[:, None] - positions[None, :], 0.0)
        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_bias = (1.0 - causal) * -1e9
        p = tf.nn.softplus(self.kerple_log_p[layer_idx])
        a = tf.nn.softplus(self.kerple_log_a[layer_idx])
        log_decay = -p * tf.math.log(1.0 + a * dist[None, :, :])
        return (causal_bias[None, :, :] + log_decay)[None, :, :, :]

    def call(self, inputs, training=None):
        interactions = inputs['interactions']
        patterns = inputs['patterns']
        features = inputs['forget_features']

        inter_e = self.inter_emb(interactions)
        pattern_e = self.pattern_emb(patterns)
        feat_e = self.feat_proj(features)

        x = tf.concat([inter_e, pattern_e, feat_e], axis=-1)
        x = self.input_proj(x)

        elapsed = features[:, :, 0:1]
        time_enc = self.time_proj(elapsed)
        seq_len = tf.shape(interactions)[1]
        positions = tf.cast(tf.range(seq_len), tf.float32)
        positions = positions[None, :, None] / tf.cast(seq_len, tf.float32)
        pos_enc = self.pos_proj(positions)

        x = x + time_enc + pos_enc
        x = self.input_dropout(x, training=training)

        for i in range(self._n_layers):
            attn_bias = self._build_attention_bias(seq_len, i)
            attn_out = self.attn_layers[i](
                x, x, attention_mask=attn_bias, training=training)
            attn_out = self.attn_dropouts[i](attn_out, training=training)
            x = self.ln1_layers[i](x + attn_out)
            ffn_out = self.ffn_layers[i](x, training=training)
            x = self.ln2_layers[i](x + ffn_out)

        return self.output_dense(x)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_csv('data/students_969.csv', parse_dates=['created_at', 'submitted_at'])

    # ── Load precomputed train/test split + IRT (avoids slow discrimination loop) ──
    train_user_ids = set(pd.read_csv('data/train_user_ids.csv')['user_id'])
    test_user_ids = set(pd.read_csv('data/test_user_ids.csv')['user_id'])
    q_irt_legacy = pd.read_csv('data/question_irt_train.csv')
    q_irt_2pl = pd.read_csv('data/question_irt_2pl.csv')

    # ── Merge IRT & feature engineering ──────────────────────────────────
    for c in ['difficulty_logit', 'discrimination']:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.merge(q_irt_legacy, on='question_id', how='left')
    df['difficulty_logit'] = df['difficulty_logit'].fillna(0.0).clip(-5, 5)
    df['discrimination'] = df['discrimination'].fillna(0).clip(lower=0)

    # Merge 2PL IRT parameters for Kalman mastery tracker
    irt_2pl_cols = q_irt_2pl[['question_id', 'discrimination_2pl', 'difficulty_2pl']].copy()
    df = df.merge(irt_2pl_cols, on='question_id', how='left')
    df['discrimination_2pl'] = df['discrimination_2pl'].fillna(1.0).clip(0.1, 5.0)
    df['difficulty_2pl'] = df['difficulty_2pl'].fillna(0.0).clip(-5, 5)

    # EAP student ability (replaces logit(accuracy) proxy)
    from irt_fitting import compute_running_eap
    df = df.sort_values(['user_id', 'created_at']).reset_index(drop=True)
    df['student_ability_logit'] = compute_running_eap(df, q_irt_2pl, n_quad=41)

    # Encoders — fit on ALL data first so indices match the trained model
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

    # Normalization from TRAIN-only data (no leakage)
    train_mask = df['user_id'].isin(train_user_ids)
    cont_mean = df.loc[train_mask, feature_cols].mean()
    cont_std = df.loc[train_mask, feature_cols].std().replace(0, 1)

    # ── Select 100 diverse students (unique quizzes, varied accuracy) ────
    user_questions = df.groupby('user_id')['question_id'].apply(frozenset)
    from collections import Counter
    qset_counts = Counter(user_questions.values)
    most_common_qset = qset_counts.most_common(1)[0][0]
    shared_quiz_users = set(user_questions[user_questions == most_common_qset].index)

    user_stats = (df.groupby('user_id')
                  .agg(n=('is_correct', 'size'), acc=('is_correct', 'mean'))
                  .reset_index())
    # Prefer unique-quiz students, fill with shared-quiz
    unique_users = user_stats[~user_stats['user_id'].isin(shared_quiz_users)].copy()
    shared_users = user_stats[user_stats['user_id'].isin(shared_quiz_users)].copy()

    # Pick evenly across accuracy range
    unique_users = unique_users.sort_values('acc')
    shared_users = shared_users.sort_values('acc')
    n_unique = min(70, len(unique_users))
    n_shared = min(100 - n_unique, len(shared_users))

    pick_u = np.linspace(0, len(unique_users) - 1, n_unique, dtype=int) if n_unique > 0 else []
    pick_s = np.linspace(0, len(shared_users) - 1, n_shared, dtype=int) if n_shared > 0 else []
    selected_ids = set(unique_users.iloc[pick_u]['user_id']) | set(shared_users.iloc[pick_s]['user_id'])

    # Filter to selected students only
    df = df[df['user_id'].isin(selected_ids)].copy()

    n_skills = int(df['subject_idx'].max() + 1)  # use max+1 (not nunique) to match model embedding

    uid_to_idx = dict(zip(df['user_id'], df['user_idx']))
    test_user_idxs = set(uid_to_idx[uid] for uid in test_user_ids if uid in uid_to_idx)

    subjects = sorted(df['subject'].unique())
    taxonomy = {s: s for s in subjects}
    user_idx_to_id = dict(zip(df['user_idx'], df['user_id']))
    subject_idx_to_name = dict(df.drop_duplicates('subject_idx')[['subject_idx', 'subject']].values)
    subject_idx_to_name = {int(k): v for k, v in subject_idx_to_name.items()}

    return df, n_skills, n_patterns, feature_cols, test_user_idxs, taxonomy, user_idx_to_id, cont_mean, cont_std, subject_idx_to_name


@st.cache_resource(show_spinner="Loading SAINT-Lite model...")
def load_model(n_skills, n_patterns, n_features):
    model = SAINTLite(n_skills, n_patterns, n_features,
                      embed_dim=56, n_heads=4, n_layers=2, ff_dim=128, dropout=0.2)
    dummy = {
        'interactions': tf.zeros((1, 10), dtype='int32'),
        'patterns': tf.zeros((1, 10), dtype='int32'),
        'forget_features': tf.zeros((1, 10, n_features), dtype='float32'),
    }
    model(dummy, training=False)
    model.load_weights('models/saint_subject_969_v3_fold0.weights.h5')
    return model


def get_student_predictions(model, df, feature_cols, cont_mean, cont_std, user_idx):
    """Return ALL interactions with SAINT-Lite predictions via batched inference.

    SAINT-Lite is subject-level: it outputs Dense(n_skills) logits per position.
    We select the target subject's logit and sigmoid it to get P(correct).
    Position 0 always gets 0.5 (no history).
    """
    sort_col = 'submitted_at' if 'submitted_at' in df.columns else 'created_at'
    ud = df[df['user_idx'] == user_idx].sort_values(sort_col)
    if len(ud) < 2:
        return None, None

    sk = ud['subject_idx'].values
    co = ud['correct'].values
    pa = ud['pattern_idx'].values
    n_f = len(feature_cols)
    ff = (ud[feature_cols].fillna(0).values - cont_mean.values) / cont_std.values
    ia = sk * 2 + co  # interaction tokens
    n = len(sk)
    max_seq = 200
    preds = np.full(n, 0.5)  # position 0 stays 0.5

    if n <= max_seq:
        # Single chunk — one forward pass
        seq_len = n - 1
        ip = ia[:seq_len].reshape(1, seq_len)
        pp = pa[:seq_len].reshape(1, seq_len)
        fp = ff[:seq_len].reshape(1, seq_len, n_f)
        logits = model({'interactions': ip, 'patterns': pp, 'forget_features': fp},
                       training=False).numpy()
        tgt = sk[1:n]
        oh = np.eye(logits.shape[-1])[tgt]
        preds[1:n] = np.clip(1.0 / (1.0 + np.exp(-np.sum(logits[0] * oh, axis=-1))), 0.01, 0.99)
    else:
        # Collect all chunks, then run as a single batch
        stride = max_seq // 2
        chunks = []  # (start, end) pairs
        chunks.append((0, max_seq))
        s = stride
        while s + max_seq <= n:
            chunks.append((s, s + max_seq))
            s += stride
        if chunks[-1][1] < n:
            chunks.append((n - max_seq, n))

        # Build batched input — all chunks padded to same seq_len (max_seq - 1)
        B = len(chunks)
        seq_len = max_seq - 1
        ip = np.zeros((B, seq_len), dtype='int32')
        pp = np.zeros((B, seq_len), dtype='int32')
        fp = np.zeros((B, seq_len, n_f), dtype='float32')
        for i, (cs, ce) in enumerate(chunks):
            ip[i] = ia[cs:ce - 1]
            pp[i] = pa[cs:ce - 1]
            fp[i] = ff[cs:ce - 1]

        # Single batched forward pass
        all_logits = model({'interactions': ip, 'patterns': pp, 'forget_features': fp},
                           training=False).numpy()  # (B, seq_len, n_skills)

        # Fill predictions from first chunk
        tgt0 = sk[1:max_seq]
        oh0 = np.eye(all_logits.shape[-1])[tgt0]
        preds[1:max_seq] = np.clip(1.0 / (1.0 + np.exp(-np.sum(all_logits[0] * oh0, axis=-1))), 0.01, 0.99)
        covered_up_to = max_seq

        # Fill from remaining chunks (only the new positions)
        for i in range(1, B):
            cs, ce = chunks[i]
            fill_from = covered_up_to
            if fill_from >= ce:
                continue
            offset = fill_from - cs - 1
            tgt = sk[cs + 1:ce]
            oh = np.eye(all_logits.shape[-1])[tgt]
            chunk_probs = np.clip(1.0 / (1.0 + np.exp(-np.sum(all_logits[i] * oh, axis=-1))), 0.01, 0.99)
            if offset < len(chunk_probs):
                preds[fill_from:ce] = chunk_probs[offset:]
            covered_up_to = ce

    return ud.reset_index(drop=True), preds


def get_all_subject_predictions(model, df, feature_cols, cont_mean, cont_std,
                                user_idx, subject_idx_to_name):
    """Get SAINT-Lite P(correct) for ALL subjects from the student's last position.

    The model outputs Dense(n_skills) logits at every position. This function
    extracts all logits from the final position — giving the model's current
    belief about the student across every subject simultaneously.
    """
    sort_col = 'submitted_at' if 'submitted_at' in df.columns else 'created_at'
    ud = df[df['user_idx'] == user_idx].sort_values(sort_col)
    if len(ud) < 2:
        return {v: 0.5 for v in subject_idx_to_name.values()}

    sk = ud['subject_idx'].values
    co = ud['correct'].values
    pa = ud['pattern_idx'].values
    n_f = len(feature_cols)
    ff = (ud[feature_cols].fillna(0).values - cont_mean.values) / cont_std.values
    ia = sk * 2 + co
    n = len(sk)
    max_seq = 200

    # Use the last chunk (most recent context)
    if n <= max_seq:
        seq_len = n - 1
        ip = ia[:seq_len].reshape(1, seq_len)
        pp = pa[:seq_len].reshape(1, seq_len)
        fp = ff[:seq_len].reshape(1, seq_len, n_f)
    else:
        start = n - max_seq
        seq_len = max_seq - 1
        ip = ia[start:start + seq_len].reshape(1, seq_len)
        pp = pa[start:start + seq_len].reshape(1, seq_len)
        fp = ff[start:start + seq_len].reshape(1, seq_len, n_f)

    logits = model({'interactions': ip, 'patterns': pp, 'forget_features': fp},
                   training=False).numpy()  # (1, seq_len, n_skills)

    # Extract ALL subject logits from the last position
    last_logits = logits[0, -1, :]  # (n_skills,)
    all_probs = np.clip(1.0 / (1.0 + np.exp(-last_logits)), 0.01, 0.99)

    return {subject_idx_to_name[i]: float(all_probs[i])
            for i in range(len(all_probs))
            if i in subject_idx_to_name}


# ═══════════════════════════════════════════════════════════════════════
# PROCESS ONE STUDENT (cached)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def process_student(_model, _df, _feature_cols, _cont_mean, _cont_std,
                    _taxonomy, user_idx, user_id):
    """Run full pipeline for one student. Returns all results needed for display."""
    interactions_df, predictions = get_student_predictions(
        _model, _df, _feature_cols, _cont_mean, _cont_std, user_idx)
    if interactions_df is None:
        return None

    real_uid = str(user_id)
    pipeline = MultiVelocityPipeline(_taxonomy)

    ts0 = interactions_df['submitted_at'].iloc[0].timestamp() / 86400.0

    # Track per-interaction velocities for the timeline chart
    velocity_timeline = {
        'interaction': [], 'subject': [], 'is_correct': [], 'kt_prediction': [],
        'timestamp_days': [],
        'kt': [], 'mastery_delta': [], 'ensemble': [],
        'cumulative_accuracy': [],
    }
    running_correct = 0

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

        running_correct += int(is_correct)
        velocity_timeline['interaction'].append(i + 1)
        velocity_timeline['subject'].append(topic)
        velocity_timeline['is_correct'].append(is_correct)
        velocity_timeline['kt_prediction'].append(kt_pred)
        velocity_timeline['timestamp_days'].append(ts_days)
        velocity_timeline['kt'].append(result['velocities']['kt'])
        velocity_timeline['mastery_delta'].append(result['velocities']['mastery_delta'])
        velocity_timeline['ensemble'].append(result['velocities']['ensemble'])
        velocity_timeline['cumulative_accuracy'].append(running_correct / (i + 1))

    current_ts = interactions_df['submitted_at'].iloc[-1].timestamp() / 86400.0 - ts0
    mvs_all = pipeline.get_mvs_all(real_uid, current_ts)

    # Per-subject breakdown
    user_all = _df[_df['user_idx'] == user_idx].sort_values('submitted_at')
    subject_data = []
    for subj, grp in user_all.groupby('subject'):
        corr = grp['is_correct'].values
        if len(corr) < 4:
            continue
        mid = len(corr) // 2
        first_acc = corr[:mid].mean()
        second_acc = corr[mid:].mean()
        actual_delta = second_acc - first_acc
        n = len(corr)

        subject_data.append({
            'Subject': subj,
            'N': n,
            '1st Half Acc': first_acc,
            '2nd Half Acc': second_acc,
            'Actual Delta': actual_delta,
            'KT Logit': pipeline.trackers['kt'].get_subject_velocity(real_uid, subj),
            'Mastery \u0394': pipeline.trackers['mastery_delta'].get_subject_velocity(real_uid, subj),
            'Ensemble': pipeline.ensemble.get_subject_velocity(real_uid, subj),
        })

    # Overall improvement
    all_corr = user_all['is_correct'].values
    mid_all = len(all_corr) // 2
    improvement = float(all_corr[mid_all:].mean() - all_corr[:mid_all].mean()) if mid_all >= 5 else 0.0

    # Build question-level sample table — 10 questions spread across subjects
    sample_indices = []
    subjects_in_data = interactions_df['subject'].unique()
    per_subj = max(1, 10 // len(subjects_in_data))
    for subj in subjects_in_data:
        subj_idxs = interactions_df.index[interactions_df['subject'] == subj].tolist()
        # Pick evenly spaced indices from each subject
        if len(subj_idxs) <= per_subj:
            sample_indices.extend(subj_idxs)
        else:
            step = len(subj_idxs) / per_subj
            sample_indices.extend([subj_idxs[int(i * step)] for i in range(per_subj)])
    # Trim to 10, sorted by interaction order
    sample_indices = sorted(sample_indices)[:10]

    sample_rows = []
    for idx in sample_indices:
        row = interactions_df.iloc[idx]
        sample_rows.append({
            '#': idx + 1,
            'Subject': row['subject'],
            'Question': str(row.get('question_id', ''))[:12] + '...',
            'Actual': 'Correct' if bool(row['correct']) else 'Wrong',
            'P(correct)': float(predictions[idx]),
            'Predicted': 'Correct' if predictions[idx] >= 0.5 else 'Wrong',
            'Match': 'Yes' if (bool(row['correct']) == (predictions[idx] >= 0.5)) else 'No',
            'Difficulty': float(row.get('difficulty_logit', 0.0)),
        })

    return {
        'velocity_timeline': pd.DataFrame(velocity_timeline),
        'mvs_all': mvs_all,
        'subject_data': pd.DataFrame(subject_data),
        'question_sample': pd.DataFrame(sample_rows),
        'n_interactions': len(interactions_df),
        'overall_accuracy': user_all['is_correct'].mean(),
        'improvement': improvement,
        'n_subjects': user_all['subject'].nunique(),
        'pipeline': pipeline,
        'real_uid': real_uid,
    }


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATE METRICS (cached)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Computing aggregate metrics...")
def compute_aggregate_metrics(_model, _df, _feature_cols, _cont_mean, _cont_std,
                              _taxonomy, _user_idx_to_id, test_user_idxs_tuple):
    test_user_idxs = set(test_user_idxs_tuple)
    approach_names = ['kt', 'mastery_delta', 'ensemble']
    student_results = []
    subject_results = []

    for uid in sorted(test_user_idxs):
        interactions_df, predictions = get_student_predictions(
            _model, _df, _feature_cols, _cont_mean, _cont_std, uid)
        if interactions_df is None:
            continue

        real_uid = str(_user_idx_to_id[uid])
        pipeline = MultiVelocityPipeline(_taxonomy)
        ts0 = interactions_df['submitted_at'].iloc[0].timestamp() / 86400.0

        for i in range(len(interactions_df)):
            row = interactions_df.iloc[i]
            topic = row['subject']
            is_correct = bool(row['correct'])
            ts_days = row['submitted_at'].timestamp() / 86400.0 - ts0
            kt_pred = float(predictions[i])
            disc_2pl = float(row.get('discrimination_2pl', 1.0))
            diff_2pl = float(row.get('difficulty_2pl', 0.0))
            pipeline.process_interaction(real_uid, topic, is_correct, ts_days, kt_pred,
                                         discrimination=disc_2pl, difficulty=diff_2pl)

        current_ts = interactions_df['submitted_at'].iloc[-1].timestamp() / 86400.0 - ts0
        mvs_all = pipeline.get_mvs_all(real_uid, current_ts)

        user_all = _df[_df['user_idx'] == uid].sort_values('submitted_at')
        all_corr = user_all['is_correct'].values
        mid_all = len(all_corr) // 2
        actual_improvement = float(all_corr[mid_all:].mean() - all_corr[:mid_all].mean()) if mid_all >= 5 else np.nan

        student_row = {
            'user_idx': uid, 'n_interactions': len(interactions_df),
            'overall_accuracy': user_all['is_correct'].mean(),
            'actual_improvement': actual_improvement,
        }
        for approach in approach_names:
            student_row[f'{approach}_velocity'] = mvs_all[approach]['velocity_raw']
            student_row[f'{approach}_velocity_norm'] = mvs_all[approach]['velocity_normalized']
            student_row[f'{approach}_consistency'] = mvs_all[approach]['consistency']
            student_row[f'{approach}_mvs'] = mvs_all[approach]['mvs']
        student_results.append(student_row)

        for subj, grp in user_all.groupby('subject'):
            corr = grp['is_correct'].values
            if len(corr) < 4:
                continue
            mid = len(corr) // 2
            subj_row = {
                'subject': subj,
                'n_interactions': len(corr),
                'actual_improvement': corr[mid:].mean() - corr[:mid].mean(),
            }
            for name, tracker in pipeline.trackers.items():
                subj_row[f'{name}_velocity'] = tracker.get_subject_velocity(real_uid, subj)
            subj_row['ensemble_velocity'] = pipeline.ensemble.get_subject_velocity(real_uid, subj)
            subject_results.append(subj_row)

    students_df = pd.DataFrame(student_results)
    subjects_df = pd.DataFrame(subject_results)

    metrics = {}
    for approach in approach_names:
        m = {}
        valid_students = students_df.dropna(subset=['actual_improvement'])
        valid_subjects = subjects_df.dropna(subset=['actual_improvement'])

        if len(valid_students) >= 5:
            v = valid_students[f'{approach}_velocity'].values
            imp = valid_students['actual_improvement'].values
            if np.std(v) > 1e-10 and np.std(imp) > 1e-10:
                pr, pp = pearsonr(v, imp)
                sr, _ = spearmanr(v, imp)
            else:
                pr, pp, sr = 0.0, 1.0, 0.0
            m['pearson_student'] = pr
            m['pearson_student_p'] = pp
            m['spearman_student'] = sr
        else:
            m['pearson_student'] = m['pearson_student_p'] = m['spearman_student'] = np.nan

        if len(valid_subjects) >= 5:
            v = valid_subjects[f'{approach}_velocity'].values
            imp = valid_subjects['actual_improvement'].values
            if np.std(v) > 1e-10 and np.std(imp) > 1e-10:
                pr, pp = pearsonr(v, imp)
                sr, _ = spearmanr(v, imp)
            else:
                pr, pp, sr = 0.0, 1.0, 0.0
            m['pearson_subject'] = pr
            m['pearson_subject_p'] = pp
            m['spearman_subject'] = sr
        else:
            m['pearson_subject'] = m['pearson_subject_p'] = m['spearman_subject'] = np.nan

        m['mean_consistency'] = students_df[f'{approach}_consistency'].mean()
        m['velocity_mean'] = students_df[f'{approach}_velocity'].mean()
        m['velocity_std'] = students_df[f'{approach}_velocity'].std()
        metrics[approach] = m

    return metrics, len(students_df)


# ═══════════════════════════════════════════════════════════════════════
# ZPD/ITZS helpers now imported from mastery_velocity
# (zpd_score, review_urgency_sigmoid, information_gain, constants)


# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════

df, n_skills, n_patterns, feature_cols, test_user_idxs, taxonomy, user_idx_to_id, cont_mean, cont_std, subject_idx_to_name = load_data()
model = load_model(n_skills, n_patterns, len(feature_cols))

n_students = df['user_id'].nunique()
n_interactions = len(df)
n_subjects = len(taxonomy)

# ─── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("KT Velocity Dashboard")
st.sidebar.caption(f"{n_students} students | {n_interactions:,} interactions | {n_subjects} subjects | SAINT-Lite")

# Build student dropdown with info
user_stats = (df.groupby('user_id')
              .agg(n=('is_correct', 'size'), acc=('is_correct', 'mean'))
              .sort_values('n', ascending=False)
              .reset_index())

student_options = []
for _, row in user_stats.iterrows():
    uid_short = str(row['user_id'])[:8]
    label = f"{uid_short}... | acc={row['acc']:.0%} | n={row['n']:.0f}"
    student_options.append((label, row['user_id']))

selected_label = st.sidebar.selectbox(
    "Select Student",
    options=[s[0] for s in student_options],
    index=4,
)
selected_user_id = student_options[[s[0] for s in student_options].index(selected_label)][1]
selected_user_idx = df[df['user_id'] == selected_user_id]['user_idx'].iloc[0]

st.sidebar.divider()
show_aggregate = st.sidebar.checkbox("Show Aggregate Metrics", value=False,
                                      help="Runs all test students (~60s)")

# ─── Process selected student ────────────────────────────────────────

result = process_student(
    model, df, feature_cols, cont_mean, cont_std,
    taxonomy, selected_user_idx, selected_user_id)

if result is None:
    st.error("Not enough data for this student (need >= 2 interactions)")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# TABS: Dashboard + Model Card
# ═══════════════════════════════════════════════════════════════════════

tab_dashboard, tab_model = st.tabs(["Dashboard", "Model Card"])

with tab_dashboard:

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1: Student Overview
    # ═══════════════════════════════════════════════════════════════════════

    st.header("Student Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interactions", f"{result['n_interactions']}",
                help="Total number of questions this student has answered.")
    col2.metric("Accuracy", f"{result['overall_accuracy']:.1%}",
                help="Percentage of questions answered correctly across all interactions.")
    col3.metric("Improvement", f"{result['improvement']:+.1%}",
                delta=f"{result['improvement']:+.1%}",
                delta_color="normal",
                help="2nd-half accuracy minus 1st-half accuracy. "
                     "Positive means the student is doing better over time.")
    col4.metric("Subjects", f"{result['n_subjects']}",
                help="Number of distinct UPSC subjects this student has practiced.")

    # Cumulative accuracy chart
    vt = result['velocity_timeline']
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=vt['interaction'], y=vt['cumulative_accuracy'],
        mode='lines', name='Cumulative Accuracy',
        line=dict(color='#2563eb', width=2),
    ))
    for i, row in vt.iterrows():
        color = 'rgba(34,197,94,0.08)' if row['is_correct'] else 'rgba(239,68,68,0.08)'
        fig_acc.add_vrect(x0=row['interaction']-0.5, x1=row['interaction']+0.5,
                          fillcolor=color, layer='below', line_width=0)
    fig_acc.update_layout(
        title="Cumulative Accuracy Over Time",
        xaxis_title="Interaction #", yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300, margin=dict(t=40, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    st.caption("**How to read this:** The blue line shows running accuracy (total correct / total attempted). "
               "Green/red background stripes show individual correct/wrong answers. "
               "A rising line means the student is improving; flat means stable performance.")


    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2: SAINT-Lite Model — Actual vs Predicted
    # ═══════════════════════════════════════════════════════════════════════

    st.header("SAINT-Lite Model: Actual vs Predicted")
    st.caption("The SAINT-Lite model predicts P(correct) for each question. "
               "These charts compare its rolling predictions against actual outcomes.")

    vt_display = vt.copy()
    W = 20  # rolling window

    # Exclude position 0 (default 0.5 prediction, no real model output)
    kt_preds = vt_display['kt_prediction'].values[1:]
    actuals = vt_display['is_correct'].astype(int).values[1:]
    try:
        student_auc = roc_auc_score(actuals, kt_preds)
    except ValueError:
        student_auc = float('nan')
    student_brier = brier_score_loss(actuals, kt_preds)
    student_acc = (actuals == (kt_preds >= 0.5).astype(int)).mean()

    # ── Chart 1: Rolling predicted vs actual ──
    rolling_actual = vt_display['is_correct'].astype(float).rolling(W, min_periods=5).mean()
    rolling_pred = vt_display['kt_prediction'].rolling(W, min_periods=5).mean()

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(
        x=vt_display['interaction'], y=rolling_actual,
        mode='lines', name=f'Actual (rolling {W})',
        line=dict(color='#2563eb', width=2.5),
    ))
    fig_rolling.add_trace(go.Scatter(
        x=vt_display['interaction'], y=rolling_pred,
        mode='lines', name=f'Predicted (rolling {W})',
        line=dict(color='#7c3aed', width=2.5),
    ))
    # Shade the gap between them
    fig_rolling.add_trace(go.Scatter(
        x=list(vt_display['interaction']) + list(vt_display['interaction'][::-1]),
        y=list(rolling_actual.fillna(0.5)) + list(rolling_pred.fillna(0.5)[::-1]),
        fill='toself', fillcolor='rgba(124,58,237,0.08)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    fig_rolling.update_layout(
        xaxis_title="Interaction #", yaxis_title="Accuracy / P(correct)",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=320, margin=dict(t=30, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ── Chart 2: Calibration plot ──
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_actual, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (kt_preds >= bin_edges[i]) & (kt_preds < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_actual.append(actuals[mask].mean())
            bin_counts.append(mask.sum())

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Perfect',
        line=dict(color='#9ca3af', width=1, dash='dash'), showlegend=False,
    ))
    fig_cal.add_trace(go.Bar(
        x=bin_centers, y=bin_counts, name='Count', yaxis='y2',
        marker_color='rgba(37,99,235,0.15)', width=0.08, showlegend=False,
    ))
    fig_cal.add_trace(go.Scatter(
        x=bin_centers, y=bin_actual, mode='lines+markers', name='Model',
        line=dict(color='#7c3aed', width=2.5),
        marker=dict(size=8, color='#7c3aed'),
    ))
    fig_cal.update_layout(
        xaxis_title="Predicted P(correct)", yaxis_title="Actual % Correct",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        yaxis2=dict(overlaying='y', side='right', showgrid=False, showticklabels=False),
        height=320, margin=dict(t=30, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ── Layout: charts + metrics ──
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.plotly_chart(fig_rolling, use_container_width=True)
        st.caption(f"**Rolling {W}-question window.** Blue = actual accuracy, Purple = model's average prediction. "
                   "When lines overlap, the model tracks the student well. Gap = prediction error.")
    with col2:
        st.plotly_chart(fig_cal, use_container_width=True)
        st.caption("**Calibration plot.** Each dot shows: of questions where the model predicted ~X%, "
                   "how many were actually correct? Points on the diagonal = perfectly calibrated.")
    with col3:
        st.metric("AUC", f"{student_auc:.3f}" if not np.isnan(student_auc) else "N/A",
                  help="Area Under ROC Curve. 1.0 = perfect, 0.5 = random.")
        st.metric("Brier Score", f"{student_brier:.3f}",
                  help="Mean squared error of probability predictions. Lower is better. 0 = perfect.")
        st.metric("Accuracy", f"{student_acc:.1%}",
                  help="% of interactions where the model's binary prediction matches reality.")

    # Question-level sample table
    st.subheader("Sample: 10 Questions Across Subjects")
    st.caption("Concrete examples showing what the SAINT-Lite model predicted vs what actually happened. "
               "This proves the model is making real, per-subject predictions — not just using averages.")

    qs = result['question_sample']
    if not qs.empty:
        def color_match(val):
            if val == 'Yes':
                return 'background-color: rgba(34,197,94,0.2); color: #4ade80'
            elif val == 'No':
                return 'background-color: rgba(239,68,68,0.2); color: #f87171'
            return ''

        def color_actual(val):
            if val == 'Correct':
                return 'color: #4ade80; font-weight: bold'
            return 'color: #f87171; font-weight: bold'

        qs_styled = qs.style.format({
            'P(correct)': '{:.1%}',
            'Difficulty': '{:+.2f}',
        }).map(color_match, subset=['Match']).map(color_actual, subset=['Actual', 'Predicted'])

        st.dataframe(qs_styled, use_container_width=True, hide_index=True)

        n_match = (qs['Match'] == 'Yes').sum()
        st.caption(f"Model got **{n_match}/{len(qs)}** correct on these questions. "
                   f"P(correct) is the model's confidence — e.g., 0.72 means \"72% chance of getting it right.\"")


    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2.5: Mastery State (EKF)
    # ═══════════════════════════════════════════════════════════════════════

    st.header("Mastery State (KT-Fused Extended Kalman Filter)")
    st.caption("The KT-Fused EKF combines SAINT-Lite's cross-subject predictions with "
               "the Extended Kalman Filter's per-topic ability tracking. It jointly estimates "
               "ability (theta) and learning rate (alpha) per subject. At each interaction, "
               "alpha is folded into theta via the state transition (theta += alpha), so "
               "mastery = sigmoid(theta) already reflects the cumulative learning trend. "
               "Confidence grows with more observations and shrinks after long gaps (FSRS forgetting).")

    pipeline = result.get('pipeline')
    if pipeline and hasattr(pipeline, 'base_pipeline'):
        mastery_tracker = pipeline.base_pipeline.mastery
        real_uid = result['real_uid']

        if hasattr(mastery_tracker, 'students') and real_uid in mastery_tracker.students:
            student_states = mastery_tracker.students[real_uid]
            last_ts = result['velocity_timeline']['timestamp_days'].max()

            # Build per-subject mastery table by aggregating topic states
            subject_mastery_rows = []
            subject_histories = {}  # {subject: [(ts, mastery), ...]}

            # Group topics by subject
            topic_to_subject = mastery_tracker.taxonomy if hasattr(mastery_tracker, 'taxonomy') else {}
            subject_topics = {}
            for topic_id, subject_id in topic_to_subject.items():
                subject_topics.setdefault(subject_id, []).append(topic_id)

            for subject, topics in sorted(subject_topics.items()):
                subj_thetas = []
                subj_alphas = []
                subj_masteries = []
                subj_confidences = []
                subj_n = 0
                subj_last_ts = 0.0
                subj_history = []

                for topic in topics:
                    if topic not in student_states:
                        continue
                    ts_state = student_states[topic]
                    if ts_state.n_interactions == 0:
                        continue
                    subj_thetas.append(ts_state.theta)
                    subj_alphas.append(ts_state.alpha)
                    m = mastery_tracker.get_topic_mastery(real_uid, topic, last_ts)
                    c = mastery_tracker.get_topic_confidence(real_uid, topic, last_ts)
                    subj_masteries.append(m)
                    subj_confidences.append(c)
                    subj_n += ts_state.n_interactions
                    subj_last_ts = max(subj_last_ts, ts_state.last_timestamp)

                    # Collect history for chart (confidence-weighted mastery)
                    for h_entry in ts_state.history:
                        h_ts, h_m = h_entry[0], h_entry[1]
                        h_c = h_entry[2] if len(h_entry) > 2 else 0.5
                        # Pull uncertain values toward prior (0.5)
                        h_m_weighted = h_m * h_c + 0.5 * (1.0 - h_c)
                        subj_history.append((h_ts, h_m_weighted))

                if subj_n == 0:
                    continue

                # Confidence-weighted means
                weights = np.array(subj_confidences)
                if weights.sum() > 0:
                    w_norm = weights / weights.sum()
                    avg_theta = float(np.sum(np.array(subj_thetas) * w_norm))
                    avg_alpha = float(np.sum(np.array(subj_alphas) * w_norm))
                    avg_mastery = float(np.sum(np.array(subj_masteries) * w_norm))
                    avg_confidence = float(np.mean(subj_confidences))
                else:
                    avg_theta = float(np.mean(subj_thetas))
                    avg_alpha = float(np.mean(subj_alphas))
                    avg_mastery = float(np.mean(subj_masteries))
                    avg_confidence = 0.0

                days_ago = max(0.0, last_ts - subj_last_ts)

                subject_mastery_rows.append({
                    'Subject': subject,
                    'theta': avg_theta,
                    'alpha': avg_alpha,
                    'Mastery': avg_mastery,
                    'Confidence': avg_confidence,
                    'N': subj_n,
                    'Last Practiced': f"{days_ago:.1f}d ago" if days_ago > 0.01 else "just now",
                })

                # Sort history by timestamp for chart
                if subj_history:
                    subj_history.sort(key=lambda x: x[0])
                    subject_histories[subject] = subj_history

            if subject_mastery_rows:
                mastery_df = pd.DataFrame(subject_mastery_rows)

                def color_mastery(val):
                    if val >= 0.7:
                        return 'background-color: rgba(34,197,94,0.2); color: #4ade80'
                    elif val <= 0.4:
                        return 'background-color: rgba(239,68,68,0.2); color: #f87171'
                    return 'background-color: rgba(234,179,8,0.15); color: #fbbf24'

                def color_alpha(val):
                    if val > 0.01:
                        return 'color: #4ade80; font-weight: bold'
                    elif val < -0.01:
                        return 'color: #f87171; font-weight: bold'
                    return 'color: #9ca3af'

                styled_mastery = mastery_df.style.format({
                    'theta': '{:+.3f}',
                    'alpha': '{:+.4f}',
                    'Mastery': '{:.3f}',
                    'Confidence': '{:.3f}',
                }).map(color_mastery, subset=['Mastery']).map(color_alpha, subset=['alpha'])

                st.dataframe(styled_mastery, use_container_width=True, hide_index=True)
                st.caption(
                    "**theta** = ability (logit scale, 0 = average). "
                    "**alpha** = learning rate (positive = improving, folded into theta each step). "
                    "**Mastery** = sigmoid(theta) — theta already includes alpha's cumulative effect. Green >= 0.70, red <= 0.40. "
                    "**Confidence** = 1 / (1 + sqrt(variance)), grows with data, shrinks with gaps."
                )

                # Mastery evolution chart
                if subject_histories:
                    fig_mastery = go.Figure()
                    colors = ['#2563eb', '#7c3aed', '#ea580c', '#16a34a', '#dc2626',
                              '#d97706', '#0891b2', '#be185d', '#4f46e5', '#059669']

                    # Determine time span to pick the right unit
                    all_ts = [h[0] for hist in subject_histories.values() for h in hist]
                    max_span_days = max(all_ts) - min(all_ts) if all_ts else 0
                    if max_span_days < 2:
                        time_scale, time_label = 24.0, "Hours Since First Interaction"
                    else:
                        time_scale, time_label = 1.0, "Days Since First Interaction"

                    for i, (subj, history) in enumerate(sorted(subject_histories.items())):
                        timestamps = [h[0] * time_scale for h in history]
                        masteries = [h[1] for h in history]
                        fig_mastery.add_trace(go.Scatter(
                            x=timestamps, y=masteries,
                            mode='lines+markers', name=subj,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=3),
                        ))
                    fig_mastery.add_hline(y=0.5, line_dash="dash", line_color="gray",
                                          opacity=0.4, annotation_text="Prior (0.5)")
                    fig_mastery.update_layout(
                        title="Mastery Evolution Over Time (EKF)",
                        xaxis_title=time_label,
                        yaxis_title="Mastery",
                        yaxis=dict(range=[0, 1], tickformat='.0%'),
                        height=400, margin=dict(t=40, b=40, l=50, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_mastery, use_container_width=True)
                    st.caption(
                        "Each point is a Kalman filter update after the student answers a question "
                        "in that subject. The EKF adjusts mastery based on question difficulty "
                        "(harder questions move the estimate more) and accounts for forgetting "
                        "during gaps between practice sessions."
                    )
            else:
                st.info("No mastery data available for this student.")
        else:
            st.info("Mastery tracker has no data for this student.")
    else:
        st.info("Pipeline not available.")


    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3: Velocity Dashboard
    # ═══════════════════════════════════════════════════════════════════════

    st.header("Learning Velocity")
    st.caption("Velocity measures **how fast** a student is improving (or declining). "
               "The **Ensemble** combines KT and Mastery Delta — both difficulty-aware signals — with adaptive confidence weighting.")

    # Ensemble first (primary metric)
    ens_mvs = result['mvs_all']['ensemble']
    st.subheader("Ensemble Velocity")
    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
    ecol1.metric("Velocity", f"{ens_mvs['velocity_raw']:+.4f}",
                 help="Confidence-weighted combination of KT and Mastery Delta (both difficulty-aware). "
                      "KT dominates early (cold start), Mastery Delta dominates with 16+ interactions.")
    ecol2.metric("MVS", f"{ens_mvs['mvs']:.1f}",
                 help="Mastery-Velocity Score from the ensemble. "
                      "Combines: mastery level, velocity, consistency, and breadth.")
    ecol3.metric("Consistency", f"{ens_mvs['consistency']:.3f}",
                 help="Weighted average of component tracker consistencies.")
    ecol4.metric("V Normalized", f"{ens_mvs['velocity_normalized']:.3f}",
                 help="Velocity mapped to [0,1] for MVS calculation. 0.5 = neutral.")

    with st.expander("How MVS is calculated"):
        st.markdown(r"""
    **Mastery-Velocity Score (MVS)** — a single 0-100 summary combining four signals:

    $$\text{MVS} = 100 \times (0.40 \times M + 0.30 \times V_{\text{norm}} + 0.15 \times C + 0.15 \times B)$$

    | Component | Weight | Range | Meaning |
    |-----------|--------|-------|---------|
    | **Mastery** ($M$) | 40% | [0, 1] | Confidence-weighted average mastery across studied subjects |
    | **Velocity** ($V_{\text{norm}}$) | 30% | [0, 1] | Raw velocity mapped to [0,1] via sigmoid. 0.5 = neutral, >0.5 = improving |
    | **Consistency** ($C$) | 15% | [0, 1] | Stability of velocity signal. High = steady progress, low = erratic |
    | **Breadth** ($B$) | 15% | [0, 1] | Fraction of subjects attempted: $\min(1, n_{\text{subjects}} / n_{\text{total}})$ |

    **Example:** A student with mastery=0.65, velocity_norm=0.58, consistency=0.72, breadth=0.50:
    MVS = 100 × (0.40×0.65 + 0.30×0.58 + 0.15×0.72 + 0.15×0.50) = **62.7**

    ---

    **Why these weights (40/30/15/15)?**

    - **Mastery (40%)**: The strongest single indicator of exam readiness. A student at 0.8 mastery
      is objectively further along than one at 0.3, regardless of velocity. Mastery receives the
      highest weight because it directly reflects what the student *knows right now*.
    - **Velocity (30%)**: Captures trajectory — the most actionable signal for intervention. A student
      at 0.5 mastery but improving fast (v > 0) is on a better path than one stagnating at 0.6.
      Second-highest weight because trajectory predicts where mastery *will be* at exam time.
    - **Consistency (15%)**: Rewards steady progress over erratic swings. A student who improves then
      crashes then improves is less prepared than a steady improver, even at the same average velocity.
      Lower weight because it's a modifier on velocity, not an independent signal.
    - **Breadth (15%)**: UPSC requires coverage across all subjects. A student acing 3/16 subjects is
      less prepared than one with moderate mastery across 12/16. Same weight as consistency because
      it's similarly a secondary signal — important but not primary.
    """)

    st.divider()
    st.subheader("Component Trackers")

    approaches = [
        ('kt', 'KT Logit', '#7c3aed'),
        ('mastery_delta', 'Mastery \u0394', '#0d9488'),
    ]
    approach_label_map = {k: label for k, label, _ in approaches}
    approach_label_map['ensemble'] = 'Ensemble'

    cols = st.columns(2)
    for i, (key, label, color) in enumerate(approaches):
        mvs = result['mvs_all'][key]
        with cols[i]:
            st.markdown(f"**{label}**")
            st.metric("Velocity", f"{mvs['velocity_raw']:+.4f}",
                       help={
                           'kt': "KT P(\u0394): Change in SAINT-Lite model's P(correct) between consecutive questions, "
                                        "smoothed with EMA. Uses the neural network's view of student progress.",
                           'mastery_delta': "Mastery \u0394: EMA-smoothed change in Kalman mastery per interaction. "
                                            "Directly reflects EKF's estimate of learning progress.",
                       }[key])
            st.metric("MVS", f"{mvs['mvs']:.1f}",
                       help="Mastery-Velocity Score (0-100). Combines: mastery level (how much they know), "
                            "velocity (how fast they're improving), consistency (how stable), "
                            "and breadth (how many subjects studied).")
            st.metric("Consistency", f"{mvs['consistency']:.3f}",
                       help="Stability of the velocity signal (0-1). High means the student is "
                            "improving steadily. Low means erratic — good in some sessions, bad in others.")


    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4: Per-Subject Velocity Table
    # ═══════════════════════════════════════════════════════════════════════

    st.header("Per-Subject Breakdown")
    st.caption("How is velocity distributed across subjects? This table splits each subject's interactions "
               "into first half and second half, then compares the **actual** accuracy delta with "
               "what each velocity algorithm detected. Green = positive (improving), red = negative (declining).")

    if not result['subject_data'].empty:
        sdf = result['subject_data'].copy()

        def color_velocity(val):
            if val > 0.01:
                return 'color: #4ade80'
            elif val < -0.01:
                return 'color: #f87171'
            return 'color: #9ca3af'

        styled = sdf.style.format({
            '1st Half Acc': '{:.1%}', '2nd Half Acc': '{:.1%}',
            'Actual Delta': '{:+.3f}',
            'KT Logit': '{:+.4f}',
            'Mastery \u0394': '{:+.4f}', 'Ensemble': '{:+.4f}',
        }).map(color_velocity, subset=['Actual Delta', 'KT Logit', 'Mastery \u0394', 'Ensemble'])

        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption("**Actual Delta** = 2nd half accuracy - 1st half accuracy. "
                   "A good velocity algorithm should be positive when Actual Delta is positive, "
                   "and negative when it's negative.")
    else:
        st.info("Not enough per-subject data (need >= 4 interactions per subject)")


    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5: Velocity Over Time
    # ═══════════════════════════════════════════════════════════════════════

    st.header("Velocity Over Time")
    st.caption("How does the velocity signal evolve as the student answers more questions? "
               "The zero line (dashed) means no change — above zero is improvement, below is decline.")

    fig_vel = go.Figure()
    # Ensemble first (thicker line, prominent color)
    fig_vel.add_trace(go.Scatter(
        x=vt['interaction'], y=vt['ensemble'],
        mode='lines', name='Ensemble',
        line=dict(color='#059669', width=3),
    ))
    for key, label, color in approaches:
        fig_vel.add_trace(go.Scatter(
            x=vt['interaction'], y=vt[key],
            mode='lines', name=label,
            line=dict(color=color, width=1.5, dash='dot'),
            opacity=0.7,
        ))
    fig_vel.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_vel.update_layout(
        xaxis_title="Interaction #", yaxis_title="Velocity",
        height=400, margin=dict(t=30, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_vel, use_container_width=True)
    st.caption("**Ensemble** (green, solid) is the confidence-weighted combination of KT and Mastery Delta (both difficulty-aware). "
               "Component trackers shown as dotted lines: "
               "**KT Logit** (purple) tracks the SAINT-Lite model's view, "
               "**Mastery \u0394** (teal) tracks EKF mastery changes.")


    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5: Algorithm Explanations
    # ═══════════════════════════════════════════════════════════════════════

    st.header("Algorithm Explanations")

    with st.expander("Ensemble Velocity", expanded=True):
        st.markdown(r"""
    **Formula:**

    $$v_{\text{ensemble}} = \frac{\sum_k w_k \cdot m_k(n) \cdot c_k \cdot \tilde{v}_k}{\sum_k w_k \cdot m_k(n) \cdot c_k}$$

    Where:
    - $w_k$: base weight (KT=0.400, Mastery \u0394=0.600)
    - $m_k(n)$: adaptive multiplier based on interaction count $n$
    - $c_k$: tracker self-reported confidence (ramps with data availability)
    - $\tilde{v}_k$: velocity scaled to [-1, +1]

    **Adaptive multipliers by data regime:**

    | Interactions | KT | Mastery \u0394 | Rationale |
    |:---:|:---:|:---:|:---|
    | n < 4 | **2.0** | 1.0 | Cold start: KT works from 2nd interaction |
    | 4-7 | **1.5** | 1.2 | Warming up: KT still leads |
    | 8-15 | 1.0 | 1.0 | Balanced: both trackers at full weight |
    | 16+ | 0.8 | **1.3** | Rich data: Mastery Delta dominates (EKF converged) |

    **Why it works:**
    Both components are **difficulty-aware** — they account for question difficulty and discrimination:
    - **KT (40%)**: EMA-smoothed change in SAINT-Lite P(correct). Trained on IRT features, so getting a
      hard question wrong causes a smaller velocity drop than a raw binary signal would. Fast cold start.
    - **Mastery Delta (60%)**: EMA-smoothed change in EKF mastery (sigmoid(theta)). Uses 2PL IRT difficulty,
      discrimination, and FSRS forgetting curve. The most principled velocity signal.

    ---

    **Why 60/40 (Mastery Delta / KT) and not 50/50?**

    The two signals measure different things and have complementary strengths:

    | Property | Mastery Delta (60%) | KT (40%) |
    |----------|:---:|:---:|
    | Observation model | 2PL IRT (separates ability from difficulty) | Neural network (learned from data) |
    | Forgetting | FSRS temporal decay built in | No explicit forgetting |
    | Uncertainty | Kalman gain adapts (high early, low late) | Fixed EMA weight |
    | Cold start | Needs ~8 interactions for EKF convergence | Works from 2nd interaction |
    | Cross-subject | Per-topic only | Captures cross-subject correlations |

    Mastery Delta gets 60% because it is the more **principled** signal — it tracks changes in the
    EKF's Bayesian posterior, which already integrates question difficulty, discrimination, temporal
    forgetting, and uncertainty. When the EKF has converged (n >= 16), its velocity signal is the
    most reliable.

    KT gets 40% because it provides two advantages Mastery Delta lacks: **(1) fast cold start** — it
    produces useful velocity from the 2nd interaction, while EKF needs ~8 interactions to converge,
    and **(2) cross-subject transfer** — SAINT-Lite captures correlations between subjects (r=0.82),
    so improvement in Polity can signal improvement in related Economy topics.

    The **adaptive multipliers** handle the regime transitions: KT dominates early (2.0x at cold start),
    Mastery Delta dominates late (1.3x with rich data). The base 60/40 split reflects the steady-state
    relative value of each signal when both have sufficient data.

    """)

    with st.expander("Mastery Delta (60% of ensemble)"):
        st.markdown(r"""
    **Formula:**

    $$v(t) = \alpha \cdot (\text{mastery}(t) - \text{mastery}(t-1)) + (1 - \alpha) \cdot v(t-1)$$

    Where $\alpha = 0.3$ (EMA smoothing) and $\text{mastery}(t) = \sigma(\theta_t)$ from the Extended Kalman Filter.

    **How it works:**
    1. The EKF maintains a latent ability estimate $\theta$ per topic, updated via 2PL IRT observations
    2. After each interaction, mastery = $\sigma(\theta)$ changes based on the outcome, question difficulty,
       and question discrimination
    3. Take the difference in mastery between consecutive interactions for the same subject
    4. Apply EMA smoothing to reduce noise

    **Range:** approximately [-0.07, +0.07] raw, scaled to [-1, +1] by the ensemble (SF = 15).

    **Worked example:**
    Polity mastery after interactions: [0.500, 0.538, 0.524, 0.561, 0.589]
    Raw deltas: [+0.038, -0.014, +0.037, +0.028]
    EMA (alpha=0.3): [+0.011, -0.004, +0.008, +0.014]
    **v = +0.014** (steady improvement in EKF mastery)

    **Why it's the strongest signal:**
    - **Difficulty-aware**: Getting a hard question wrong barely moves mastery; getting it right moves it a lot.
      Conversely, easy questions have little impact either way. This is because the 2PL IRT observation model
      weights updates by question discrimination and difficulty.
    - **Forgetting-aware**: The EKF's predict step applies FSRS temporal decay. A student who hasn't practiced
      in a while sees mastery drop, which the velocity tracker captures as negative velocity.
    - **Uncertainty-aware**: The Kalman gain adapts — early interactions move mastery more (high uncertainty),
      later interactions are more stable (low uncertainty). This naturally smooths the velocity signal.
    """)

    with st.expander("KT Logit Derivative (40% of ensemble)"):
        st.markdown(r"""
    **Formula:**

    $$v(t) = \alpha \cdot (P_{kt}(t) - P_{kt}(t-1)) + (1 - \alpha) \cdot v(t-1)$$

    Where $\alpha = 0.3$ (EMA smoothing) and $P_{kt}$ is the SAINT-Lite model's P(correct) prediction.

    **How it works:**
    1. For each interaction, the SAINT-Lite model produces P(correct)
    2. Take the difference between consecutive predictions for the same subject
    3. Apply EMA smoothing to reduce noise

    **Range:** approximately [-0.5, +0.5].

    **Worked example:**
    KT predictions for History: [0.45, 0.52, 0.48, 0.55, 0.60]
    Raw deltas: [+0.07, -0.04, +0.07, +0.05]
    EMA (alpha=0.3): [+0.021, -0.003, +0.019, +0.024]
    **v = +0.024** (moderate improvement in KT model's eyes)

    **Key advantage:** Already difficulty-controlled (KT model sees IRT features).
    **Key limitation:** Measures prediction change, not actual learning.

    ---

    **Why 40% and not more?**

    KT velocity is a useful complement but has two limitations that keep it below 50%:
    1. **No explicit forgetting model**: If a student hasn't practiced History in 2 weeks, KT's last
       prediction is stale. Mastery Delta's EKF applies FSRS decay automatically.
    2. **No uncertainty tracking**: KT always applies the same EMA weight (alpha=0.3), regardless of
       whether it's the 3rd or 300th interaction. Mastery Delta adapts via the Kalman gain.

    Its 40% weight reflects its two unique strengths: **fast cold start** (useful velocity from the
    2nd interaction) and **cross-subject transfer** (SAINT-Lite captures r=0.82 correlations
    between subjects). The adaptive multipliers boost KT to 80% effective weight at cold start
    (n < 4) where these advantages matter most.
    """)




    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 6: Recommendation Scoring (v2)
    # ═══════════════════════════════════════════════════════════════════════

    st.header("Recommendation Scoring (ITZS)")

    st.markdown(rf"""
    **ITZS (Information-Theoretic ZPD Scoring)** — 4 orthogonal components:

    $$\text{{score}} = w_e \cdot \text{{ELG}} + w_r \cdot \text{{review}} + w_i \cdot \text{{info\_gain}} + w_n \cdot \text{{novelty}}$$

    | Component | Weight | Formula | Source |
    |-----------|--------|---------|--------|
    | Expected Learning Gain | {ITZS_WEIGHTS['w_elg']:.2f} | $\text{{zpd}}(P_{{kt}}) \times (1 - mastery)$ | Wilson 2019 + Clément 2015 |
    | Review Urgency | {ITZS_WEIGHTS['w_review']:.2f} | $\sigma(-{REVIEW_SIGMOID_K:.0f} \cdot (R - {REVIEW_R_THRESHOLD:.2f}))$ | FSRS v5 |
    | Information Gain | {ITZS_WEIGHTS['w_info']:.2f} | $H(P_{{kt}}) \times (1 - mastery)$ | Shannon entropy |
    | Novelty Bonus | {ITZS_WEIGHTS['w_novelty']:.2f} | $\text{{zpd}}(P_{{kt}}) \times \max(0, 1 - n/{NOVELTY_DECAY_N})$ | Exploration |

    **ZPD score** (85% rule, Wilson et al. 2019):
    $\text{{zpd}}(p) = \exp\left(-\frac{{(p - {ZPD_PEAK:.4f})^2}}{{2 \times {ZPD_SIGMA}^2}}\right)$

    Peak at $P(\text{{correct}}) = 1 - \frac{{\text{{erfc}}(1/\sqrt{{2}})}}{{2}} \approx {ZPD_PEAK:.4f}$ — analytically derived
    optimal accuracy for gradient-descent learners (not a heuristic 0.85).

    **Information Gain** (Shannon entropy): $H(p) = -p \log_2 p - (1-p) \log_2(1-p)$
    Drives exploration toward topics where the KT model is most uncertain ($p \approx 0.5$).
    """)

    # Show component curves
    col_zpd, col_sigmoid, col_entropy = st.columns(3)

    with col_zpd:
        p_range = np.linspace(0, 1, 200)
        zpd_vals = [zpd_score(p) for p in p_range]
        fig_zpd = go.Figure()
        fig_zpd.add_trace(go.Scatter(x=p_range, y=zpd_vals, mode='lines',
                                      line=dict(color='#2563eb', width=2)))
        fig_zpd.add_vline(x=ZPD_PEAK, line_dash="dash", line_color="red",
                           annotation_text=f"Peak: P={ZPD_PEAK:.4f}")
        fig_zpd.update_layout(title="ZPD Score (85% Rule)",
                               xaxis_title="P(correct)", yaxis_title="ZPD Score",
                               height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_zpd, use_container_width=True)

    with col_sigmoid:
        r_range = np.linspace(0, 1, 200)
        sigmoid_vals = [review_urgency_sigmoid(r) for r in r_range]
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=r_range, y=sigmoid_vals, mode='lines',
                                      line=dict(color='#ea580c', width=2)))
        fig_sig.add_vline(x=REVIEW_R_THRESHOLD, line_dash="dash", line_color="red",
                           annotation_text=f"Threshold: R={REVIEW_R_THRESHOLD:.2f}")
        fig_sig.update_layout(title="Review Urgency Sigmoid",
                               xaxis_title="Retrievability R", yaxis_title="Urgency",
                               height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_sig, use_container_width=True)

    with col_entropy:
        p_range_e = np.linspace(0.01, 0.99, 200)
        entropy_vals = [information_gain(p) for p in p_range_e]
        fig_ent = go.Figure()
        fig_ent.add_trace(go.Scatter(x=p_range_e, y=entropy_vals, mode='lines',
                                      line=dict(color='#16a34a', width=2)))
        fig_ent.add_vline(x=0.5, line_dash="dash", line_color="red",
                           annotation_text="Max: P=0.50")
        fig_ent.update_layout(title="Information Gain H(p)",
                               xaxis_title="P(correct)", yaxis_title="Entropy (bits)",
                               height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_ent, use_container_width=True)

    with st.expander("Why these ITZS weights?"):
        st.markdown(r"""
    The four ITZS components address **orthogonal goals** — each captures a different reason
    to recommend a topic. Their weights reflect priority ordering:

    | Component | Weight | Why this weight |
    |-----------|--------|-----------------|
    | **Expected Learning Gain** | 35% | **Highest** because learning efficiency is the primary goal. Targets the 85% optimal accuracy (Wilson et al. 2019), where the gradient of the learning curve is steepest. A topic in the ZPD with low mastery yields the most improvement per interaction. |
    | **Review Urgency** | 25% | **Second** because forgetting is the biggest threat to mastery. Uses FSRS v5 retrievability (trained on 350M flashcard reviews) to detect memory decay. Without review, a topic at R < 0.70 will be forgotten — all prior learning wasted. |
    | **Information Gain** | 15% | **Third** because reducing uncertainty about student ability improves all future recommendations. Shannon entropy is maximized at P = 0.50 (maximum uncertainty). Topics here are most informative for the mastery tracker. |
    | **Novelty Bonus** | 10% | **Lowest** because exploration must be balanced against exploitation. Decays linearly to zero after N interactions per topic. Prevents the system from only recommending familiar topics while keeping exploitation dominant. |

    The weights sum to 0.85 (the remaining 0.15 was previously allocated to Exam Importance, which has been
    removed since PYQ frequency data is not yet integrated).

    **Design principle:** ELG and Review Urgency together (60%) ensure the system alternates between
    "learn new material at the optimal difficulty" and "reinforce what's about to be forgotten."
    Information Gain and Novelty (25%) add strategic exploration so the system doesn't get stuck
    in a narrow topic loop.
    """)

    # Show scored recommendations for this student
    st.subheader("Recommendations for This Student")

    # Get SAINT-Lite P(correct) for ALL subjects from model's last position
    pipeline = result.get('pipeline')
    rec_data = []
    if pipeline:
        real_uid = result['real_uid']
        kt_by_subject = get_all_subject_predictions(
            model, df, feature_cols, cont_mean, cont_std,
            selected_user_idx, subject_idx_to_name)

        # Use the pipeline's ITZS scoring (single source of truth)
        last_ts = vt['timestamp_days'].max() if 'timestamp_days' in vt.columns else None
        topic_scores = score_topics_for_recommendation(
            real_uid,
            pipeline.base_pipeline.mastery,
            last_ts,
            kt_predictions=kt_by_subject,
        )

        for ts in topic_scores:
            rec_data.append({
                'Subject': ts.topic_id,
                'Mastery': ts.mastery,
                'P(kt)': ts.kt_prediction,
                'R': ts.retrievability,
                'Review Urg.': ts.review_urgency,
                'ZPD': ts.zpd_score_val,
                'ELG': ts.expected_learning_gain,
                'Info Gain': ts.information_gain_val,
                'Novelty': ts.novelty_bonus,
                'Total': ts.total_score,
                'N': ts.n_interactions,
            })

    rec_df = pd.DataFrame(rec_data)
    if not rec_df.empty:
        st.dataframe(
            rec_df.style.format({
                'Mastery': '{:.2f}', 'P(kt)': '{:.2f}', 'R': '{:.2f}',
                'Review Urg.': '{:.3f}', 'ZPD': '{:.3f}', 'ELG': '{:.3f}',
                'Info Gain': '{:.3f}', 'Novelty': '{:.3f}',
                'Total': '{:.3f}',
            }).background_gradient(subset=['Total'], cmap='YlOrRd'),
            use_container_width=True, hide_index=True,
        )

        top3 = rec_df.head(3)['Subject'].tolist()
        st.success(f"Top 3 recommended subjects: **{', '.join(top3)}**")

        # Show which topics benefit from KT vs mastery-only
        with st.expander("KT vs Mastery-Only Comparison"):
            st.markdown("""
            The table above uses **SAINT-Lite predictions** for P(correct) where available.
            Topics without KT predictions fall back to mastery as the P(correct) estimate.
            """)
            kt_diff = []
            for ts in topic_scores:
                kt_p = kt_by_subject.get(ts.topic_id)
                if kt_p is not None:
                    kt_diff.append({
                        'Subject': ts.topic_id,
                        'SAINT P(correct)': kt_p,
                        'Mastery Estimate': ts.mastery,
                        'Difference': kt_p - ts.mastery,
                        'ZPD with SAINT': zpd_score(kt_p),
                        'ZPD with Mastery': zpd_score(ts.mastery),
                    })
            if kt_diff:
                diff_df = pd.DataFrame(kt_diff)
                st.dataframe(diff_df.style.format({
                    'SAINT P(correct)': '{:.3f}', 'Mastery Estimate': '{:.3f}',
                    'Difference': '{:+.3f}', 'ZPD with SAINT': '{:.3f}',
                    'ZPD with Mastery': '{:.3f}',
                }), use_container_width=True, hide_index=True)
                st.caption("When SAINT-Lite and mastery disagree, the ZPD scores differ — "
                           "SAINT-Lite provides a more accurate difficulty estimate from the "
                           "transformer's attention over the full interaction history.")
    else:
        st.warning("No pipeline data available for recommendations.")


    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 7: Summary Comparison (aggregate)
    # ═══════════════════════════════════════════════════════════════════════

    if show_aggregate:
        st.header("Aggregate Metrics (Test Students)")

        metrics, n_test = compute_aggregate_metrics(
            model, df, feature_cols, cont_mean, cont_std,
            taxonomy, user_idx_to_id, tuple(sorted(test_user_idxs)))

        approach_labels = {
            'kt': 'KT Logit Deriv',
            'mastery_delta': 'Mastery Delta',
            'ensemble': 'Ensemble',
        }

        agg_data = []
        for key in ['ensemble', 'kt', 'mastery_delta']:
            m = metrics[key]
            sig_student = '*' if m.get('pearson_student_p', 1) < 0.05 else ''
            sig_subject = '*' if m.get('pearson_subject_p', 1) < 0.05 else ''
            agg_data.append({
                'Approach': approach_labels[key],
                'Pearson r (student)': f"{m['pearson_student']:+.3f}{sig_student}",
                'Spearman r (student)': f"{m['spearman_student']:+.3f}",
                'Pearson r (subject)': f"{m['pearson_subject']:+.3f}{sig_subject}",
                'Spearman r (subject)': f"{m['spearman_subject']:+.3f}",
                'Consistency': f"{m['mean_consistency']:.3f}",
                'Velocity Mean': f"{m['velocity_mean']:+.3f}",
                'Velocity Std': f"{m['velocity_std']:+.3f}",
            })

        st.dataframe(pd.DataFrame(agg_data), use_container_width=True, hide_index=True)

        best_subj = max(((k, v) for k, v in metrics.items() if k in approach_labels),
                        key=lambda x: x[1]['pearson_subject'])
        st.success(f"Best subject-level Pearson r: **{approach_labels[best_subj[0]]}** "
                   f"(r = {best_subj[1]['pearson_subject']:+.3f})")

# ═══════════════════════════════════════════════════════════════════════
# MODEL CARD TAB
# ═══════════════════════════════════════════════════════════════════════

# Training curve data (sampled from saint_topic_969 training log)
TRAINING_CURVE_EPOCHS = [
    1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
    85, 90, 95, 100,
]
TRAINING_CURVE_LOSS = [
    0.6188, 0.5505, 0.5359, 0.5273, 0.5164, 0.5019, 0.4791, 0.4425,
    0.4140, 0.3903, 0.3712, 0.3579, 0.3481, 0.3427, 0.3314, 0.3244,
    0.3158, 0.3111, 0.3067, 0.2981, 0.2920,
]
TRAINING_CURVE_VAL_LOSS = [
    0.5890, 0.5600, 0.5450, 0.5310, 0.5130, 0.4920, 0.4600, 0.4210,
    0.3960, 0.3780, 0.3640, 0.3540, 0.3470, 0.3420, 0.3350, 0.3300,
    0.3250, 0.3220, 0.3200, 0.3170, 0.3150,
]

with tab_model:
    st.header("Model Card: SAINT-Lite Knowledge Tracing")

    # ── Architecture ──────────────────────────────────────────────────
    st.subheader("Architecture")
    st.markdown("""
**Why Transformer over LSTM?**
- **Direct attention** to any past interaction — no information bottleneck through a fixed-size hidden state. The model can directly look at what happened 50 questions ago.
- **Monotonic decay bias** — learned per-head decay parameters (softplus gamma) automatically weight recent interactions more. This replaces fixed positional encoding with a data-driven forgetting curve.
- **Scales with data** — On 286 students, all architectures (GRU, LSTM, Transformer) hit the same ceiling. On 969 students, the transformer leverages more data significantly better than RNNs.
""")

    st.code("""
SAINT-Lite Architecture

Input per timestep:
  Interaction Token (skill * 2 + correct)  → Embedding(56d)
  Question Pattern (13 types)              → Embedding(14d)     → Concat → Dense(56d)
  11 Continuous Features                   → Dense(14d)                     ↓
                                                                    + Time Encoding (tanh)
                                                                    + Position Encoding
                                                                           ↓
  2x Transformer Block:
    Multi-Head Attention (4 heads, key_dim=14, causal mask + KerpleLog decay)
    → LayerNorm → FFN(128, GELU) → LayerNorm
                                                                           ↓
  Dense(16) → sigmoid → P(correct) per subject
                           ↓
             Extended Kalman Filter (per topic)
             State: [theta, alpha] + covariance
             Observation: 2PL IRT → mastery + confidence
""", language=None)

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Parameters", "63,062")
    mc2.metric("Embed Dim", "56")
    mc3.metric("Layers", "2")
    mc4.metric("Heads", "4")
    mc5.metric("FF Dim", "128")
    mc6.metric("Dropout", "0.2")

    # ── Dataset ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Dataset")
    st.caption("UPSC practice platform. Students with >= 10 interactions selected. "
               "Train/test split stratified by overall accuracy.")

    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    dc1.metric("Students", "969")
    dc2.metric("Interactions", "110,314")
    dc3.metric("Questions", "1,977")
    dc4.metric("Subjects", "16")
    dc5.metric("Topics", "53")

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Train", "775 students (80%)")
    tc2.metric("Test", "194 students (20%)")
    tc3.metric("Sequence Length", "200 max")

    # ── Training Curves ───────────────────────────────────────────────
    st.divider()
    st.subheader("Training Curves")

    fig_train = go.Figure()
    fig_train.add_trace(
        go.Scatter(x=TRAINING_CURVE_EPOCHS, y=TRAINING_CURVE_LOSS,
                   mode='lines+markers', name='Train Loss',
                   line=dict(color='#ef4444', width=2),
                   marker=dict(size=4)))
    fig_train.add_trace(
        go.Scatter(x=TRAINING_CURVE_EPOCHS, y=TRAINING_CURVE_VAL_LOSS,
                   mode='lines+markers', name='Val Loss',
                   line=dict(color='#2563eb', width=2),
                   marker=dict(size=4)))
    fig_train.update_layout(
        height=400, margin=dict(t=30, b=40, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_train.update_xaxes(title_text="Epoch")
    fig_train.update_yaxes(title_text="Loss", range=[0.2, 0.65])

    st.plotly_chart(fig_train, use_container_width=True)
    st.caption("Both train and validation loss decrease rapidly and converge by epoch 100. "
               "The narrow gap between train and val loss confirms the model is not overfitting. "
               "Early stopping (patience=10) selects the best checkpoint. "
               "Training config: lr=1e-3, batch_size=8, AdamW, warmup=3 epochs.")

    # ── Subject Correlations ──────────────────────────────────────────
    st.divider()
    st.subheader("Subject Correlations")
    st.markdown("""
Subject accuracies are **highly correlated** (mean Pearson r=0.82, PCA PC1=85%).
A single "general UPSC ability" factor explains most of the variance across subjects.
But this general factor **collapses on hard questions** — on the hardest quintile,
subjects become nearly independent (r=0.17, PC1=29%).
""")

    corr_data = pd.DataFrame({
        'Difficulty': ['Very Easy (Q1)', 'Easy (Q2)', 'Medium (Q3)', 'Hard (Q4)', 'Very Hard (Q5)', 'Overall'],
        'Accuracy': ['88%', '73%', '63%', '51%', '33%', '65%'],
        'Mean r': [0.337, 0.575, 0.443, 0.397, 0.174, 0.817],
        'PCA PC1': ['43%', '64%', '52%', '48%', '29%', '85%'],
    })
    st.dataframe(corr_data, use_container_width=True, hide_index=True)

    st.markdown("""
**How this informs the model:**
The high cross-subject correlation enables **cold-start transfer** — a student's Economy accuracy
informs Polity predictions even before they attempt Polity. The model encodes this through:
- **Interaction tokens** (`subject * 2 + correct`) — the transformer sees all subjects in the
  attention window and learns cross-subject patterns.
- **IRT student ability** — `student_ability_logit` captures general ability explicitly as a feature.
- **Subject-level output** — the Dense(16) output head predicts all subjects jointly,
  encouraging shared representations.
""")

    # ── Within-Subject Topic Correlations ─────────────────────────────
    with st.expander("Within-Subject Topic Correlations"):
        st.caption("At the topic level (53 L2 topics), the general factor disappears. "
                   "Mean topic correlation r=0.23, PCA PC1=22%, needing 20 components for 80% variance. "
                   "This confirms topic-level tracking is necessary.")
        topic_corr = pd.DataFrame({
            'Subject': ['Geography', 'Current Affairs', 'Polity', 'Environment',
                        'Economy', 'History', 'Science & Tech'],
            'Within-Subject Topic r': [0.329, 0.298, 0.288, 0.246, 0.222, 0.210, 0.193],
            'Interpretation': [
                'Spatial reasoning transfers',
                'General awareness transfers',
                'Constitutional knowledge transfers',
                'Moderate transfer',
                'Topics somewhat independent',
                'Ancient/Modern/Medieval are different skills',
                'Biotech, space, defense are distinct',
            ],
        })
        st.dataframe(topic_corr, use_container_width=True, hide_index=True)

    # ── Feature Engineering ───────────────────────────────────────────
    st.divider()
    st.subheader("Feature Engineering (11 Features)")
    st.caption("All features are z-normalized (mean=0, std=1) using training-fold statistics, "
               "then projected through Dense(14d) before concatenation with embeddings.")

    feat_data = pd.DataFrame({
        'Group': ['Time', 'Time', 'Time',
                  'Prior Perf.', 'Prior Perf.',
                  'Engagement', 'Engagement', 'Engagement',
                  'IRT', 'IRT', 'IRT'],
        'Feature': [
            'elapsed_log', 'subject_elapsed_log', 'topic_elapsed_log',
            'subject_accuracy_prior', 'topic_accuracy_prior',
            'time_spent_log', 'subject_attempts_log', 'topic_attempts_log',
            'difficulty_logit', 'discrimination', 'student_ability_logit',
        ],
        'Transform': [
            'log(1 + seconds)', 'log(1 + seconds)', 'log(1 + seconds)',
            'Raw [0,1], default 0.5', 'Raw [0,1], default 0.5',
            'log(1 + seconds)', 'log(1 + count)', 'log(1 + count)',
            '-log(acc / (1-acc))', 'point-biserial corr', 'EAP Bayesian (N(0,1) prior, 41 quad)',
        ],
        'Purpose': [
            'Time since last interaction (any subject) — forgetting proxy',
            'Time since last practice in this subject',
            'Time since last practice in this topic',
            'Running accuracy in this subject before current question',
            'Running accuracy in this topic before current question',
            'Time spent on previous question — engagement signal',
            'Number of prior attempts in this subject',
            'Number of prior attempts in this topic',
            'Question difficulty from train-only IRT (questions with >= 5 attempts)',
            'How well the question separates strong vs weak students',
            'Student general ability on the logit scale',
        ],
    })
    st.dataframe(feat_data, use_container_width=True, hide_index=True)

    with st.expander("Why these 11 features?"):
        st.markdown(r"""
    Each feature was chosen because it captures a distinct signal that **raw binary outcomes alone cannot provide**.
    The features fall into 4 groups, each addressing a different blind spot:

    **1. Time features (3) — forgetting signal**

    Binary outcomes tell you *what* the student got right, but not *when*. A correct answer
    after 2 minutes of study is very different from a correct answer after 2 weeks of no practice.

    | Feature | Why needed |
    |---------|-----------|
    | `elapsed_log` | Overall gap since last interaction — captures general forgetting and session boundaries |
    | `subject_elapsed_log` | Gap since last practice in this *subject* — captures subject-specific forgetting |
    | `topic_elapsed_log` | Gap since last practice in this *topic* — captures fine-grained skill decay |

    Three levels (overall, subject, topic) because forgetting operates at different timescales.
    A student who practiced Economy yesterday but hasn't done Polity in a week has different
    forgetting states for each. Log transform because forgetting follows a power law (FSRS v5),
    so the difference between 1 day and 2 days matters much more than 29 days vs 30 days.

    **2. Prior performance features (2) — baseline ability signal**

    | Feature | Why needed |
    |---------|-----------|
    | `subject_accuracy_prior` | Running accuracy in this subject *before* the current question — gives the model a stable baseline for how the student performs in this subject |
    | `topic_accuracy_prior` | Same but at topic level — captures specific skill gaps within a subject |

    Two levels because a student might be strong in Polity overall (75% accuracy) but weak
    in Constitutional Bodies specifically (40%). The subject prior gives a stable estimate;
    the topic prior captures finer differentiation. Default 0.5 for unseen subjects/topics
    (uninformative prior).

    **3. Engagement features (3) — effort and familiarity signal**

    | Feature | Why needed |
    |---------|-----------|
    | `time_spent_log` | Time spent on the *previous* question — proxy for engagement. A student who spends 3 seconds is guessing; one who spends 90 seconds is reasoning |
    | `subject_attempts_log` | Count of prior attempts in this subject — captures familiarity |
    | `topic_attempts_log` | Count of prior attempts in this topic — captures topic-specific practice volume |

    These features give the model context that binary outcomes miss. A student with 50%
    accuracy over 5 attempts is very different from one with 50% accuracy over 50 attempts
    (the second has converged; the first is still noisy).

    **4. IRT features (3) — question quality signal**

    | Feature | Why needed |
    |---------|-----------|
    | `difficulty_logit` | How hard is this question? Without this, the model can't distinguish "strong student answered an easy question" from "weak student answered a hard question" |
    | `discrimination` | How informative is this question? High-discrimination questions cleanly separate strong from weak students; low-discrimination questions are noisy and should be downweighted |
    | `student_ability_logit` | Student's general ability (EAP estimate). Captures the r=0.82 cross-subject correlation — a student's performance in Economy predicts their performance in Polity |

    These are the most impactful features. Difficulty alone achieves **AUC 0.717** (vs 0.702
    without it). Together, the IRT features let the model decompose P(correct) into
    student ability vs question properties — the fundamental insight of Item Response Theory.
    """)

    # ── IRT Parameter Calculation ─────────────────────────────────────
    with st.expander("How IRT Parameters Are Calculated"):
        st.markdown(r"""
**Difficulty (Rasch 1PL):** Computed from **train-only** question accuracy (questions with >= 5 attempts).

$$\text{difficulty\_logit} = -\log\left(\frac{\text{accuracy}}{1 - \text{accuracy}}\right)$$

Easy questions → large negative values (e.g., 90% accuracy → -2.20). Hard questions → positive (e.g., 20% → +1.39).

**Discrimination (IRT 2PL):** Point-biserial correlation between `is_correct` and student ability per question, computed on training students only.

$$\text{discrimination} = r_{pb}(\text{is\_correct}, \text{user\_ability})$$

High discrimination (> 0.5) = question cleanly separates strong from weak students.
Low discrimination (< 0.2) = nearly random.

**Student Ability (EAP):** Computed using **Expected A Posteriori** Bayesian estimation — numerical
quadrature over a N(0,1) prior, updated incrementally after each response using 2PL item parameters.

$$\hat{\theta}_{\text{EAP}} = \frac{\sum_q \theta_q \cdot L(\mathbf{x} \mid \theta_q) \cdot \pi(\theta_q)}{\sum_q L(\mathbf{x} \mid \theta_q) \cdot \pi(\theta_q)}$$

where $L(\mathbf{x} \mid \theta_q)$ is the likelihood of the student's response history at quadrature point $\theta_q$,
and $\pi(\theta_q)$ is the N(0,1) prior. Unlike `logit(accuracy)`, EAP correctly weights responses by question
difficulty and discrimination, giving smooth, stable ability estimates from the first interaction.

**Key insight:** Difficulty alone achieves AUC 0.717 — confirming it's a critical signal.
On the 286-student dataset, the full GRU model without difficulty only achieved AUC 0.702.
""")

    # ── Evaluation ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Evaluation")

    st.metric("Per-Student AUC", "0.896",
              help="Average AUC computed per student on held-out test data, then averaged.")
    st.caption("See the Training Curves section above for train/val loss convergence.")

    # ── Cold Start ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Cold Start")

    st.markdown("""
**The problem:** A new student has zero history — the model has nothing to condition on.

**Two mechanisms handle cold start:**

**1. IRT features — known before the student answers:**
- **`difficulty_logit`** — tells the model *how hard* this question is (easy questions have
  high base rates regardless of the student). Difficulty alone achieves **AUC 0.717**.
- **`discrimination`** — tells the model *how informative* this question's outcome is
  (high-discrimination questions are worth trusting; low-discrimination ones are noisy)
- **`student_ability_logit`** — initialized at 0 (average) for new students, then updated
  after each response

**2. General ability estimation — the implicit cold start engine:**

Our correlation analysis shows that subject scores are highly correlated (mean r = 0.82),
but this is **not** because UPSC subjects share content — it's because **85% of the variance
is explained by a single general ability factor** (PCA PC1 = 85%). A student who is strong
in Economy is likely strong in Polity too, not because the subjects overlap, but because
the same underlying preparation level drives performance everywhere.

The `student_ability_logit` feature captures this. It's computed using **EAP (Expected A Posteriori)**
Bayesian estimation — a proper IRT-based ability estimate that uses numerical quadrature over a
N(0,1) prior, updated incrementally after each response using the 2PL item parameters. Unlike the
raw logit(accuracy) proxy, EAP correctly weights responses by question difficulty and discrimination,
giving smooth, stable estimates from the very first interaction. After just a few questions in *any*
subject, this feature gives the model a reliable estimate of the student's general ability — which
predicts ~85% of their performance in every other subject, even ones they haven't touched yet.

**`COLD_START_MASK = 5`** — during training, the first 5 positions in each sequence are
excluded from the loss, so the model learns to build context before being evaluated.

**3. EKF cold-start initialization:**
The Extended Kalman Filter uses the KT model's first prediction to initialize theta:
`theta_0 = logit(P_kt)`. This gives the EKF a warm start from the neural network's
estimate before any topic-specific data exists.

After the first interaction, the KT-Fused EKF continues to incorporate KT predictions
via two mechanisms (see the KT fusion section below) — but with mathematical safeguards
against double-counting.
""")

    with st.expander("Worked example: first 3 interactions"):
        st.markdown("""
| # | Question | Difficulty | Model's Prior | Student Answers | What Updates |
|---|----------|-----------|--------------|----------------|-------------|
| 1 | Economy — easy (diff = −2.1) | 85% base rate | P(correct) ≈ 0.85 (just difficulty) | Correct | `student_ability_logit` stays ≈ 0 — easy question, not informative |
| 2 | Polity — medium (diff = −0.5) | 62% base rate | P(correct) ≈ 0.62 | Correct | `student_ability_logit` shifts up — beating a medium question is meaningful |
| 3 | History — hard (diff = +0.9) | 33% base rate | P(correct) ≈ 0.45 (adjusted up by ability) | Correct | `student_ability_logit` jumps — acing a hard question is strong signal |

After 3 interactions:
- **General ability** is now estimated above average → the model predicts higher P(correct)
  for *all* subjects, even Geography/Environment/Science which the student hasn't touched
- This works because general ability explains 85% of subject-level variance — it's not
  that Economy knowledge transfers to Geography, it's that a strong student tends to be
  strong everywhere
- **Topic-level mastery** still starts cold (topics are genuinely independent, mean r = 0.23)
  — the model only refines topic estimates as topic-specific data accumulates
""")

    # ── Mastery Estimation: Extended Kalman Filter ─────────────────────
    st.divider()
    st.subheader("Mastery Estimation: KT-Fused Extended Kalman Filter")

    st.markdown("""
**How is mastery calculated?**

Mastery is computed per topic as **sigmoid(theta)**, where theta is the student's latent ability
estimated by a KT-Fused Extended Kalman Filter. The EKF jointly tracks a 2D state vector
**[theta, alpha]** — ability and learning rate. At every interaction, the predict step applies
`theta_new = gamma * theta + alpha`, so **alpha is continuously folded into theta** before the
mastery readout. The sigmoid(theta) you see already reflects the cumulative effect of the
student's learning rate. The system has **two models** that work together:

| Model | What it does | Strengths |
|-------|-------------|-----------|
| **SAINT-Lite** (Knowledge Tracing) | Transformer that predicts P(correct) for each interaction, trained on the 11 features above | Cross-subject transfer (r=0.82), fast cold start, sees full interaction history |
| **Extended Kalman Filter** (Mastery Tracker) | Bayesian state-space model that tracks [theta, alpha] per topic with 2PL IRT updates | Explicit uncertainty, question-aware updates (difficulty + discrimination), FSRS temporal forgetting |

Neither model alone is sufficient:
- **SAINT-Lite alone** predicts P(correct) but doesn't track *mastery* — it doesn't separate student ability from question difficulty, doesn't model forgetting explicitly, and gives no uncertainty estimate.
- **EKF alone** tracks mastery rigorously but is per-topic only — it can't leverage cross-subject correlations, and ignores the rich patterns the transformer learns from the full sequence.

**The KT-Fused EKF combines both** — the EKF provides the principled Bayesian framework, while SAINT-Lite's predictions continuously inform it through two mechanisms:
""")

    st.code("""
    SAINT-Lite P(correct)                           Binary Outcome (correct/wrong)
            │                                                    │
            │                                                    │
    ┌───────┴────────────────────────────────────────────────────┴──────────┐
    │                     KT-Fused EKF (per topic)                         │
    │                                                                      │
    │  1. COLD START:  theta_0 = logit(P_kt)  ← KT initializes ability    │
    │                                                                      │
    │  2. PREDICT:     theta_pred = gamma * theta + alpha                  │
    │                  + beta_kt * delta(logit_kt)  ← KT delta nudge (B)  │
    │                                                                      │
    │  3. INJECT:      every 5 interactions, precision-weighted blend      │
    │                  of theta with KT logit (4x inflated var) (C)       │
    │                                                                      │
    │  4. UPDATE:      2PL IRT observation — binary outcome adjusts theta  │
    │                  Kalman gain adapts by uncertainty                    │
    │                                                                      │
    │  OUTPUT: mastery = sigmoid(theta)   [theta already includes alpha]    │
    │          confidence = 1/(1 + sqrt(P[0,0]))                           │
    └──────────────────────────────────────────────────────────────────────┘

    Key: KT enters the PREDICT step, binary outcomes enter the UPDATE step
         → zero double-counting (different steps use different evidence)
""", language=None)

    st.caption("Tested on 20 students (10 synthetic + 10 real): 10.9% MAE reduction vs standard EKF, "
               "13 wins, 6 losses, 1 tie, zero overconfidence.")

    with st.expander("How the Extended Kalman Filter tracks mastery", expanded=False):
        st.markdown("""
**Why an Extended Kalman Filter instead of a running average or Beta distribution?**

| Approach | Limitation |
|----------|-----------|
| **Running average** | Can't account for question difficulty (getting an easy question right != getting a hard one right), forgetting over time, or uncertainty. |
| **Beta distribution** | Tracks P(correct) directly but doesn't separate ability from question difficulty. A strong student (theta=1.0) facing a hard question (b=1.5) should have low P(correct) but high mastery — Beta conflates these. |
| **EKF with 2PL IRT** | Separates student ability (theta) from question properties (a, b). Tracks uncertainty explicitly. Accounts for temporal forgetting via FSRS. The "Extended" part handles the nonlinear sigmoid observation model. |

---

The mastery tracker uses an **Extended Kalman Filter (EKF)** — a recursive Bayesian estimator
that maintains a probabilistic belief about each student's ability per topic. Unlike simpler
approaches (running averages, Beta distributions), the EKF tracks two quantities simultaneously
and updates them optimally after every single student response.

**What the filter tracks (state vector):**

Each topic for each student has a 2D state vector:

| Component | Symbol | Meaning | Range |
|-----------|--------|---------|-------|
| **Ability** | theta | Student's latent ability on the IRT logit scale. theta = 0 means average (50% on a medium question). theta = +2 means strong (~88% on medium). theta = -2 means weak (~12% on medium). | [-6, +6] |
| **Learning rate** | alpha | How fast theta is changing per interaction. Positive = improving, negative = declining. This is the "velocity" at the mastery level. | [-0.5, +0.5] |

The filter also maintains a 2x2 **covariance matrix P** that represents uncertainty in both
theta and alpha. As the student answers more questions, P shrinks (we become more certain).
After long gaps without practice, P grows (uncertainty increases).

**The predict-update cycle:**

Every time a student answers a question, the filter runs two steps:

**Step 1 — Predict:** Before seeing the answer, project the state forward in time.
- Forgetting: theta decays toward 0 based on the FSRS retention factor gamma(dt).
  If the student hasn't practiced this topic in a while, gamma < 1 and theta shrinks
  (mastery decays). gamma comes from the FSRS power-law forgetting curve, fit to
  350 million flashcard reviews.
- Learning trend: alpha (learning rate) is added to theta. If the student has been
  improving, this predicts continued improvement.
- Uncertainty grows: process noise Q is added to the covariance, scaled by the time gap.
  Longer gaps = more uncertainty.

The state transition is:
- theta_predicted = gamma * theta_previous + alpha_previous
- alpha_predicted = alpha_previous (learning rate is assumed stable)

**Step 2 — Update:** After seeing the binary outcome (correct/wrong), correct the prediction.
- The **2PL IRT observation model** computes: P(correct) = sigmoid(a * (theta - b)), where
  a = question discrimination and b = question difficulty. This means the filter knows whether
  the question was easy or hard, and how informative the outcome is.
- The **innovation** = actual outcome (0 or 1) minus predicted probability. A positive
  innovation (correct answer when the model expected wrong) pushes theta up. Negative
  innovation pushes theta down.
- The **Kalman gain** K determines how much to trust the new observation vs the prior
  belief. When uncertainty is high (new student, long gap), K is large and the filter
  reacts strongly. When uncertainty is low (many observations, recent practice), K is
  small and the filter is stable.
- The **Jacobian** H = [a * p * (1-p), 0] linearizes the nonlinear sigmoid observation
  model. This is the "Extended" part of EKF — standard Kalman filters only handle linear
  observations.
- The **Joseph form** covariance update P = (I-KH) * P * (I-KH)^T + K * R * K^T ensures
  numerical stability even with extreme probabilities.

**Outputs:**

| Output | Formula | Meaning |
|--------|---------|---------|
| mastery | sigmoid(theta) | Probability scale [0, 1]. A student with theta=1.5 has mastery 0.82. Note: theta already incorporates alpha — the predict step adds alpha to theta at every interaction, so the learning rate is baked into the ability estimate before the sigmoid is applied. |
| confidence | 1 / (1 + sqrt(P[0,0])) | How certain we are about theta. Starts at 0.5 (prior), rises toward 1 with more observations. Falls after long gaps. Never collapses to 0. |
| learning_rate | alpha | Rate of ability change per interaction (logit-units). Estimated jointly with theta by the EKF. Positive = improving, negative = declining. Folded into theta at each predict step, and also used by velocity trackers downstream. |

**Subject mastery** is computed as the confidence-weighted mean of topic masteries within that
subject. Topics where we have more data (higher confidence) contribute more. Topics the student
hasn't practiced are excluded (they would just be the 0.5 prior, which is uninformative).

---

**KT Signal Fusion (Hybrid B+C)**

The EKF incorporates SAINT-Lite's P(correct) predictions continuously, not just at cold start.
This was validated on 20 students (10 synthetic + 10 real): **10.9% MAE reduction**, 13 wins,
6 losses, 1 tie, with zero overconfidence.

**Why fuse KT into the EKF?** The standard EKF only used KT at cold start (first interaction).
After that, the transformer's rich cross-subject predictions were completely ignored. This wastes
information — SAINT-Lite captures cross-subject correlations (r=0.82), so a student strong in
Polity gets informed ability estimates for related Economy/Governance topics.

Two fusion mechanisms work together:

**Approach B (every interaction) — KT delta as control input:**
- theta_pred = gamma * theta + alpha + **beta_kt * delta_logit_kt**
- delta_logit_kt = logit(P_kt_now) - logit(P_kt_previous), clipped to [-1, +1]
- beta_kt ramps up adaptively: beta_kt(n) = 0.3 * (1 - exp(-n/8))
- **Why it works:** KT enters the *prediction step*, binary outcomes enter the *update step*.
  Since they never compete in the same step, there's **zero double-counting** (Kalman 1960).

**Approach C (every 5 interactions) — KT level injection:**
- Precision-weighted blend of EKF theta with KT logit
- theta_fused = (prec_ekf * theta_pred + prec_kt * logit(P_kt)) / (prec_ekf + prec_kt)
- **sigma2_kt inflated to 4x EKF variance** to mitigate double-counting
- **Why it works:** Even with careful delta control, EKF and KT can drift apart over time.
  Periodic level injection gently anchors them. The 4x variance inflation ensures KT acts as
  a soft suggestion, not a hard override (Dynamic LENS, EDM 2024).
""")

    with st.expander("Journey of a Student Response", expanded=False):
        st.markdown("""
**What happens when a student answers one question — step by step:**

Suppose a student answers a **Polity** question. The question has discrimination a = 1.5
(highly informative) and difficulty b = -0.3 (slightly easy). The student's current Kalman
state for Polity is: theta = 0.4, alpha = 0.06, and it's been 0.5 days since their last
Polity question.

---

**Step 1: PREDICT — project state forward in time**

The FSRS forgetting model computes the retention factor for a 0.5-day gap:
- gamma = (1 + 0.5 / (9 * S))^(-0.5) where S = FSRS stability for this topic
- Say gamma = 0.98 (very little forgetting over 12 hours)

State prediction:
- theta_predicted = 0.98 * 0.4 + 0.06 = **0.452**
- alpha_predicted = 0.06 (unchanged)
- Covariance P grows slightly (process noise added for the 0.5-day gap)

**KT Fusion (Approach B):** The SAINT-Lite model's last P(correct) for Polity was 0.62,
now it predicts 0.67. delta_logit_kt = logit(0.67) - logit(0.62) = +0.21.
With beta_kt = 0.25 (ramped up over interactions): theta_predicted += 0.25 * 0.21 = **0.505**
(nudged up because the KT model also sees improvement).

---

**Step 2: COMPUTE predicted probability (2PL IRT)**

Using the 2PL model with the question's IRT parameters:
- z = a * (theta_predicted - b) = 1.5 * (0.452 - (-0.3)) = 1.5 * 0.752 = **1.128**
- P(correct) = sigmoid(1.128) = **0.755**

The filter expects the student to get this right with ~76% probability.

---

**Step 3: OBSERVE the answer**

The student answers **correctly** (y = 1).

Innovation = y - P(correct) = 1.0 - 0.755 = **+0.245**

This is a mildly positive surprise — the student did slightly better than expected.
If the student had gotten it wrong, the innovation would be -0.755 (a strong negative signal,
since the model expected them to get it right).

---

**Step 4: UPDATE — correct the estimate**

The Kalman gain K is computed from the ratio of prior uncertainty to total uncertainty:
- K_theta and K_alpha are computed from P_predicted, the Jacobian H, and the observation noise R
- Because discrimination is high (a=1.5), H is large, so the observation is informative
- K_theta might be ~0.15, K_alpha might be ~0.005

State update:
- theta_new = 0.452 + 0.15 * 0.245 = **0.489** (ability increases)
- alpha_new = 0.06 + 0.005 * 0.245 = **0.061** (learning rate nudges up slightly)
- P_new shrinks (we're now more certain about both theta and alpha)

---

**Step 5: COMPUTE outputs**

From the updated state:
- **mastery** = sigmoid(0.489) = **0.620** (up from 0.598 before this interaction)
- **confidence** = 1 / (1 + sqrt(P_new[0,0])) — say **0.70** (moderate, still building up)
- **learning_rate** = alpha = 0.061 (slightly positive — student trending upward)

---

**Step 6: AGGREGATE to subject and overall mastery**

The student may have practiced multiple Polity topics (e.g., Fundamental Rights, Parliament,
Constitutional Bodies). Each topic has its own Kalman state.

- **Subject mastery** for Polity = weighted average of all Polity topic masteries,
  weighted by each topic's confidence. Topics with more data count more.
- **Overall mastery** = weighted average across all subjects, weighted by
  confidence * PYQ exam importance.
- Topics the student hasn't practiced are excluded (not dragged to the 0.5 prior).

---

**Step 7: FEED into velocity and recommendations**

The mastery estimate feeds into:
1. **Velocity trackers** — KT and Mastery Delta (both difficulty-aware) combine into an Ensemble
   that detects improvement or decline
2. **MVS score** — Mastery-Velocity Score combines: 40% mastery level, 30% velocity,
   15% consistency, 15% breadth (how many subjects studied)
3. **ITZS recommendations** — Expected Learning Gain, Review Urgency, Information Gain,
   and Novelty are combined to score which topic to study next
""")

    with st.expander("From Mastery to Velocity to Recommendations (full pipeline)", expanded=False):
        st.code("""
Full Pipeline: Binary Response → Recommendation

Binary Response (correct / wrong)
  + Question IRT parameters (discrimination, difficulty)
  + Timestamp (for forgetting calculation)
                    │
                    ▼
  ┌──────────────────────────────────────────┐
  │  KT-Fused EKF (per topic)               │
  │  State: [theta, alpha] + covariance      │
  │  1. PREDICT: theta decays (FSRS)         │
  │     + KT delta control input (Approach B)│
  │  2. INJECT: KT level every 5 ixns (C)   │
  │  3. UPDATE: 2PL IRT observation          │
  │  4. OUTPUT: mastery + confidence         │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  Confidence-Weighted Aggregation    │
  │  topic mastery → subject mastery    │
  │  subject mastery → overall mastery  │
  └──────────────┬──────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  ┌───────────┐   ┌────────────────────────────────────────┐
  │  Mastery  │   │  Velocity Trackers                     │
  │  (level)  │   │  KT (40%):   SAINT delta, EMA smoothed │
  │           │   │  Mastery Δ (60%): EKF delta, EMA       │
  │           │   │           ↓                             │
  │           │   │  Ensemble: confidence-weighted combo    │
  └─────┬─────┘   └──────────────┬─────────────────────────┘
        │                        │
        └──────────┬─────────────┘
                   ▼
  ┌─────────────────────────────────────┐
  │  MVS Score (0-100)                  │
  │  = 40% mastery                      │
  │  + 30% velocity (normalized)        │
  │  + 15% consistency                  │
  │  + 15% breadth                      │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  ITZS Recommendation Scoring        │
  │  Per topic:                         │
  │    ELG     = ZPD(P_kt) × (1-m)     │
  │    Review  = σ(-k × (R - R₀))      │
  │    Info    = H(P_kt) × (1-m)       │
  │    Novelty = ZPD(P_kt) × decay(n)  │
  │                                     │
  │  Total = weighted sum → rank topics │
  └─────────────────────────────────────┘
""", language=None)
