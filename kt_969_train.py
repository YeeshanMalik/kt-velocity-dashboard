"""
Train SAINT-Lite + AKT on expanded 969-student dataset.

SAINT-Lite: Subject-level (16 skills) + Topic-level (53 skills) mastery prediction.
AKT: Question-level (per-question) prediction.

No pre-training, no synthetic data — 969 real students is enough.
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import train_test_split
from irt_fitting import compute_running_eap
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc, time

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f'TensorFlow: {tf.__version__}')
print('GPU:', tf.config.list_physical_devices('GPU'))


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING + IRT
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv('data/students_969.csv', parse_dates=['created_at', 'submitted_at'])
print(f'\nLoaded: {len(df):,} interactions, {df.user_id.nunique()} students')

# Train/test split FIRST (on user_id, before IRT)
user_stats = (df.groupby('user_id')
              .agg(mean_accuracy=('is_correct', 'mean'))
              .reset_index())
user_stats['accuracy_bin'] = pd.qcut(
    user_stats['mean_accuracy'], q=4, labels=False, duplicates='drop')
_tr_idx, _te_idx = train_test_split(
    user_stats.index, test_size=0.2, random_state=SEED,
    stratify=user_stats['accuracy_bin'])
TRAIN_USER_IDS = set(user_stats.iloc[_tr_idx]['user_id'])
TEST_USER_IDS = set(user_stats.iloc[_te_idx]['user_id'])
print(f'User split: {len(TRAIN_USER_IDS)} train, {len(TEST_USER_IDS)} test')

# Load precomputed 2PL IRT (from irt_fitting.py)
q_irt_2pl = pd.read_csv('data/question_irt_2pl.csv')
print(f'Loaded 2PL IRT: {len(q_irt_2pl)} questions')

# Use legacy columns for SAINT features (difficulty_logit, discrimination)
# These are backward-compatible with the original feature pipeline
q_irt = q_irt_2pl[['question_id', 'difficulty_logit', 'discrimination']].copy()

# Add IRT features
for c in ['difficulty_logit', 'discrimination']:
    if c in df.columns:
        df = df.drop(columns=[c])
df = df.merge(q_irt[['question_id', 'difficulty_logit', 'discrimination']],
              on='question_id', how='left')
df['difficulty_logit'] = df['difficulty_logit'].fillna(0.0).clip(-5, 5)
df['discrimination'] = df['discrimination'].fillna(0).clip(lower=0)

# EAP student ability (replaces logit(accuracy) proxy)
df = df.sort_values(['user_id', 'created_at']).reset_index(drop=True)
print('Computing running EAP theta...')
df['student_ability_logit'] = compute_running_eap(df, q_irt_2pl, n_quad=41)
print(f'EAP theta: mean={df["student_ability_logit"].mean():.3f}, '
      f'std={df["student_ability_logit"].std():.3f}')

# Encoders
encoders = {}
for col, new_col in [('user_id', 'user_idx'), ('subject', 'subject_idx')]:
    encoders[col] = LabelEncoder()
    df[new_col] = encoders[col].fit_transform(df[col].fillna('Unknown'))
df['question_pattern_clean'] = df['question_pattern'].fillna('Unknown')
encoders['pattern'] = LabelEncoder()
df['pattern_idx'] = encoders['pattern'].fit_transform(df['question_pattern_clean'])
df['L2_clean'] = df['L2'].fillna('Unknown')
encoders['topic'] = LabelEncoder()
df['topic_idx'] = encoders['topic'].fit_transform(df['L2_clean'])

# Question encoder for AKT
question_encoder = LabelEncoder()
df['question_idx'] = question_encoder.fit_transform(df['question_id'])
N_QUESTIONS = df['question_idx'].nunique()

# Rasch difficulty array for AKT
_rasch_logits = np.zeros(N_QUESTIONS + 1, dtype=np.float32)
for _, row in q_irt.iterrows():
    qid = row['question_id']
    if qid in question_encoder.classes_:
        idx = question_encoder.transform([qid])[0]
        _rasch_logits[idx] = row['difficulty_logit']
print(f'Rasch difficulty: {(np.abs(_rasch_logits) > 0).sum()} questions with IRT values')

for col, src in [('elapsed_log', 'elapsed_seconds'),
                 ('subject_elapsed_log', 'subject_elapsed_seconds'),
                 ('topic_elapsed_log', 'topic_elapsed_seconds')]:
    df[col] = np.log1p(df[src].fillna(0).clip(0, 86400 * 30))
df['subject_accuracy_prior'] = df['subject_accuracy_prior'].fillna(0.5)
df['topic_accuracy_prior'] = df['topic_accuracy_prior'].fillna(0.5)
df['time_spent_log'] = np.log1p(df['total_time_spent'].fillna(0).clip(0, 3600))
df['subject_attempts_log'] = np.log1p(df['subject_attempts_prior'].fillna(0))
df['topic_attempts_log'] = np.log1p(df['topic_attempts_prior'].fillna(0))
df['correct'] = df['is_correct'].astype(int)


# ═══════════════════════════════════════════════════════════════════════
# 2. FEATURE SETS + SPLITS
# ═══════════════════════════════════════════════════════════════════════
IRT_FEATURES = [
    'elapsed_log', 'subject_elapsed_log', 'topic_elapsed_log',
    'subject_accuracy_prior', 'topic_accuracy_prior',
    'time_spent_log', 'subject_attempts_log', 'topic_attempts_log',
    'difficulty_logit', 'discrimination', 'student_ability_logit',
]

N_SKILLS   = df['subject_idx'].nunique()
N_TOPICS   = df['topic_idx'].nunique()
N_PATTERNS = df['pattern_idx'].nunique()
N_FOLDS    = 1
COLD_START_MASK = 5
FEATURE_COLS = IRT_FEATURES
N_F = len(FEATURE_COLS)

uid_to_uidx = dict(zip(df['user_id'], df['user_idx']))
tr_users = set(uid_to_uidx[uid] for uid in TRAIN_USER_IDS if uid in uid_to_uidx)
te_users = set(uid_to_uidx[uid] for uid in TEST_USER_IDS if uid in uid_to_uidx)
folds = [(tr_users, te_users)]
print(f'Split: {len(tr_users)} train, {len(te_users)} test users')
print(f'N_QUESTIONS={N_QUESTIONS}, N_SKILLS={N_SKILLS}, N_TOPICS={N_TOPICS}, N_PATTERNS={N_PATTERNS}')
print(f'Features ({N_F}): {FEATURE_COLS}')


# ═══════════════════════════════════════════════════════════════════════
# 3. SEQUENCE BUILDING
# ═══════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return {
        'auc':      roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, (y_pred >= 0.5).astype(int)),
        'log_loss': log_loss(y_true, y_pred),
    }


# ── SAINT-Lite sequence builder (skill-level) ──
def build_saint_sequences(dataframe, user_indices, feature_cols, skill_col,
                          crop_prob=0.0, max_seq_len=None,
                          cont_mean=None, cont_std=None):
    n_f = len(feature_cols)
    if cont_mean is None:
        cont_mean = dataframe[feature_cols].mean()
    if cont_std is None:
        cont_std = dataframe[feature_cols].std().replace(0, 1)

    seqs_inter, seqs_pat, seqs_feat = [], [], []
    seqs_tsk, seqs_tgt = [], []

    for uid in user_indices:
        sort_col = 'submitted_at' if 'submitted_at' in dataframe.columns else 'created_at'
        ud = dataframe[dataframe['user_idx'] == uid].sort_values(sort_col)
        if len(ud) < 2:
            continue
        sk = ud[skill_col].values
        co = ud['correct'].values
        pa = ud['pattern_idx'].values
        ff = (ud[feature_cols].fillna(0).values - cont_mean.values) / cont_std.values
        ia = sk * 2 + co

        if crop_prob > 0 and np.random.random() < crop_prob and len(sk) > 10:
            # Train-short-test-long: always crop from START (simulates cold start)
            crop_len = np.random.randint(5, len(sk))
            sk, co, pa, ff, ia = sk[:crop_len], co[:crop_len], pa[:crop_len], ff[:crop_len], ia[:crop_len]

        seqs_inter.append(ia[:-1])
        seqs_pat.append(pa[:-1])
        seqs_feat.append(ff[:-1])
        seqs_tsk.append(sk[1:])
        seqs_tgt.append(co[1:])

    actual_max = max(len(s) for s in seqs_inter)
    pad_len = min(actual_max, max_seq_len) if max_seq_len else actual_max
    n = len(seqs_inter)

    ip  = np.zeros((n, pad_len), dtype='int32')
    pp  = np.zeros((n, pad_len), dtype='int32')
    fp  = np.zeros((n, pad_len, n_f), dtype='float32')
    tsp = np.zeros((n, pad_len), dtype='int32')
    tcp = np.zeros((n, pad_len), dtype='float32')
    mp  = np.zeros((n, pad_len), dtype='float32')

    for i in range(n):
        L = min(len(seqs_inter[i]), pad_len)
        ip[i, :L]  = seqs_inter[i][-L:]
        pp[i, :L]  = seqs_pat[i][-L:]
        fp[i, :L]  = seqs_feat[i][-L:]
        tsp[i, :L] = seqs_tsk[i][-L:]
        tcp[i, :L] = seqs_tgt[i][-L:]
        start = min(COLD_START_MASK, L)
        mp[i, start:L] = 1.0

    inputs = {'interactions': ip, 'patterns': pp, 'forget_features': fp}
    return inputs, tcp, mp, tsp


# ── AKT sequence builder (question-level) ──
def build_akt_sequences(dataframe, user_indices, feature_cols,
                        crop_prob=0.0, max_seq_len=None,
                        cont_mean=None, cont_std=None):
    n_f = len(feature_cols)
    if cont_mean is None:
        cont_mean = dataframe[feature_cols].mean()
    if cont_std is None:
        cont_std = dataframe[feature_cols].std().replace(0, 1)
    seqs_q, seqs_r, seqs_f, seqs_tgt = [], [], [], []

    for uid in user_indices:
        sort_col = 'submitted_at' if 'submitted_at' in dataframe.columns else 'created_at'
        ud = dataframe[dataframe['user_idx'] == uid].sort_values(sort_col)
        if len(ud) < 2:
            continue
        qi = ud['question_idx'].values
        co = ud['correct'].values
        ff = (ud[feature_cols].fillna(0).values - cont_mean.values) / cont_std.values

        if crop_prob > 0 and np.random.random() < crop_prob and len(qi) > 10:
            # Train-short-test-long: always crop from START (simulates cold start)
            crop_len = np.random.randint(5, len(qi))
            qi, co, ff = qi[:crop_len], co[:crop_len], ff[:crop_len]

        seqs_q.append(qi)
        seqs_r.append(co)
        seqs_f.append(ff)
        seqs_tgt.append(co[1:])

    actual_max = max(len(s) for s in seqs_q)
    pad_len = min(actual_max, max_seq_len) if max_seq_len else actual_max
    n = len(seqs_q)

    q_pad  = np.zeros((n, pad_len), dtype='int32')
    qi_pad = np.zeros((n, pad_len - 1), dtype='int32')
    r_pad  = np.zeros((n, pad_len - 1), dtype='int32')
    f_pad  = np.zeros((n, pad_len - 1, n_f), dtype='float32')
    tgt_pad = np.zeros((n, pad_len - 1), dtype='float32')
    mask_pad = np.zeros((n, pad_len - 1), dtype='float32')

    for i in range(n):
        T = min(len(seqs_q[i]), pad_len)
        q_pad[i, :T] = seqs_q[i][-T:]
        L = T - 1
        qi_pad[i, :L] = seqs_q[i][-T:-1]
        r_pad[i, :L]  = seqs_r[i][-T:-1]
        f_pad[i, :L]  = seqs_f[i][-T:-1]
        tgt_pad[i, :L] = seqs_tgt[i][-L:]
        start = min(COLD_START_MASK, L)
        mask_pad[i, start:L] = 1.0

    inputs = {
        'questions': q_pad,
        'inter_questions': qi_pad,
        'inter_responses': r_pad,
        'inter_features': f_pad,
    }
    return inputs, tgt_pad, mask_pad


# ═══════════════════════════════════════════════════════════════════════
# 4. SAINT-LITE MODEL
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
        # KerpleLog attention bias parameters (replaces exponential decay)
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
            # KerpleLog: log_decay = -p * log(1 + a * dist), per head
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
        # KerpleLog: log_decay = -p * log(1 + a * dist), logarithmic decay with heavier tails
        p = tf.nn.softplus(self.kerple_log_p[layer_idx])   # (n_heads, 1, 1)
        a = tf.nn.softplus(self.kerple_log_a[layer_idx])   # (n_heads, 1, 1)
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


def build_saint_lite(n_skills, n_patterns, n_features,
                     embed_dim=56, n_heads=4, n_layers=2, ff_dim=128, dropout=0.2):
    model = SAINTLite(n_skills, n_patterns, n_features,
                      embed_dim, n_heads, n_layers, ff_dim, dropout)
    dummy = {
        'interactions': tf.zeros((1, 10), dtype='int32'),
        'patterns': tf.zeros((1, 10), dtype='int32'),
        'forget_features': tf.zeros((1, 10, n_features), dtype='float32'),
    }
    model(dummy, training=False)
    return model


# ═══════════════════════════════════════════════════════════════════════
# 5. AKT MODEL
# ═══════════════════════════════════════════════════════════════════════

class AKTModel(keras.Model):
    def __init__(self, n_questions, n_skills, n_features,
                 embed_dim=128, n_heads=8, n_layers=4, ff_dim=256, dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_questions = n_questions
        self.n_skills = n_skills
        self.embed_dim = embed_dim
        self._n_layers = n_layers

        self.question_emb = layers.Embedding(n_questions + 1, embed_dim, mask_zero=False)
        self.diff_emb = layers.Embedding(n_questions + 1, embed_dim // 4, mask_zero=False)
        self.rasch_difficulty = layers.Embedding(
            n_questions + 1, 1, mask_zero=False,
            embeddings_initializer='zeros', trainable=False)

        self.feature_proj = layers.Dense(embed_dim // 2)
        self.response_emb = layers.Embedding(2, embed_dim // 4)
        self.inter_proj = layers.Dense(embed_dim)

        self.pos_emb_q = layers.Embedding(512, embed_dim)
        self.pos_emb_s = layers.Embedding(512, embed_dim)
        self.time_proj = layers.Dense(embed_dim, activation='tanh')
        self.input_drop = layers.Dropout(dropout)

        # Question Encoder
        self.q_self_attn, self.q_ffn = [], []
        self.q_ln1, self.q_ln2, self.q_drop, self.q_gammas = [], [], [], []
        for i in range(n_layers):
            self.q_self_attn.append(
                layers.MultiHeadAttention(n_heads, embed_dim // n_heads, dropout=dropout))
            self.q_ffn.append(keras.Sequential([
                layers.Dense(ff_dim, activation='gelu'), layers.Dropout(dropout),
                layers.Dense(embed_dim), layers.Dropout(dropout)]))
            self.q_ln1.append(layers.LayerNormalization(epsilon=1e-6))
            self.q_ln2.append(layers.LayerNormalization(epsilon=1e-6))
            self.q_drop.append(layers.Dropout(dropout))
            self.q_gammas.append(tf.Variable(
                tf.constant(-1.0, shape=(n_heads, 1, 1)), trainable=True, name=f'q_gamma_{i}'))

        # Interaction Encoder
        self.s_self_attn, self.s_ffn = [], []
        self.s_ln1, self.s_ln2, self.s_drop, self.s_gammas = [], [], [], []
        for i in range(n_layers):
            self.s_self_attn.append(
                layers.MultiHeadAttention(n_heads, embed_dim // n_heads, dropout=dropout))
            self.s_ffn.append(keras.Sequential([
                layers.Dense(ff_dim, activation='gelu'), layers.Dropout(dropout),
                layers.Dense(embed_dim), layers.Dropout(dropout)]))
            self.s_ln1.append(layers.LayerNormalization(epsilon=1e-6))
            self.s_ln2.append(layers.LayerNormalization(epsilon=1e-6))
            self.s_drop.append(layers.Dropout(dropout))
            self.s_gammas.append(tf.Variable(
                tf.constant(-1.0, shape=(n_heads, 1, 1)), trainable=True, name=f's_gamma_{i}'))

        # Knowledge Retriever
        self.kr_cross_attn, self.kr_ffn = [], []
        self.kr_ln1, self.kr_ln2, self.kr_drop, self.kr_gammas = [], [], [], []
        for i in range(n_layers):
            self.kr_cross_attn.append(
                layers.MultiHeadAttention(n_heads, embed_dim // n_heads, dropout=dropout))
            self.kr_ffn.append(keras.Sequential([
                layers.Dense(ff_dim, activation='gelu'), layers.Dropout(dropout),
                layers.Dense(embed_dim), layers.Dropout(dropout)]))
            self.kr_ln1.append(layers.LayerNormalization(epsilon=1e-6))
            self.kr_ln2.append(layers.LayerNormalization(epsilon=1e-6))
            self.kr_drop.append(layers.Dropout(dropout))
            self.kr_gammas.append(tf.Variable(
                tf.constant(-1.0, shape=(n_heads, 1, 1)), trainable=True, name=f'kr_gamma_{i}'))

        self.output_proj = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(1),
        ])

    def _causal_decay_bias(self, seq_len, log_gamma):
        positions = tf.cast(tf.range(seq_len), tf.float32)
        dist = tf.maximum(positions[:, None] - positions[None, :], 0.0)
        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_bias = (1.0 - causal) * -1e9
        gamma = tf.nn.softplus(log_gamma)
        return (causal_bias[None, :, :] + (-gamma * dist[None, :, :]))[None, :, :, :]

    def call(self, inputs, training=None):
        questions = inputs['questions']
        inter_q = inputs['inter_questions']
        inter_r = inputs['inter_responses']
        inter_f = inputs['inter_features']

        q_len = tf.shape(questions)[1]
        s_len = tf.shape(inter_q)[1]

        q_emb = self.question_emb(questions)
        q_rasch = tf.sigmoid(self.rasch_difficulty(questions))
        q_emb = q_emb * (0.5 + q_rasch)
        q_pos = self.pos_emb_q(tf.range(q_len))[None, :, :]
        q_repr = self.input_drop(q_emb + q_pos, training=training)

        iq_emb = self.question_emb(inter_q)
        iq_rasch = tf.sigmoid(self.rasch_difficulty(inter_q))
        iq_emb = iq_emb * (0.5 + iq_rasch)
        id_emb = self.diff_emb(inter_q)
        r_float = tf.cast(inter_r, tf.float32)[:, :, None]
        r_emb = self.response_emb(inter_r)
        f_emb = self.feature_proj(inter_f)

        inter_combined = tf.concat([iq_emb, id_emb * r_float, r_emb, f_emb], axis=-1)
        s_repr = self.inter_proj(inter_combined)
        s_pos = self.pos_emb_s(tf.range(s_len))[None, :, :]
        time_enc = self.time_proj(inter_f[:, :, 0:1])
        s_repr = self.input_drop(s_repr + s_pos + time_enc, training=training)

        for i in range(self._n_layers):
            bias = self._causal_decay_bias(q_len, self.q_gammas[i])
            attn = self.q_self_attn[i](q_repr, q_repr, attention_mask=bias, training=training)
            attn = self.q_drop[i](attn, training=training)
            q_repr = self.q_ln1[i](q_repr + attn)
            q_repr = self.q_ln2[i](q_repr + self.q_ffn[i](q_repr, training=training))

        for i in range(self._n_layers):
            bias = self._causal_decay_bias(s_len, self.s_gammas[i])
            attn = self.s_self_attn[i](s_repr, s_repr, attention_mask=bias, training=training)
            attn = self.s_drop[i](attn, training=training)
            s_repr = self.s_ln1[i](s_repr + attn)
            s_repr = self.s_ln2[i](s_repr + self.s_ffn[i](s_repr, training=training))

        q_query = q_repr[:, 1:, :]
        q_key = q_repr[:, :-1, :]
        s_value = s_repr

        for i in range(self._n_layers):
            bias = self._causal_decay_bias(s_len, self.kr_gammas[i])
            cross_attn = self.kr_cross_attn[i](
                query=q_query, key=q_key, value=s_value,
                attention_mask=bias, training=training)
            cross_attn = self.kr_drop[i](cross_attn, training=training)
            q_query = self.kr_ln1[i](q_query + cross_attn)
            q_query = self.kr_ln2[i](q_query + self.kr_ffn[i](q_query, training=training))

        logits = self.output_proj(q_query)
        return tf.squeeze(logits, axis=-1)


def build_akt(n_questions, n_skills, n_features,
              embed_dim=128, n_heads=8, n_layers=4, ff_dim=256, dropout=0.2,
              rasch_logits=None):
    model = AKTModel(n_questions, n_skills, n_features,
                     embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
                     ff_dim=ff_dim, dropout=dropout)
    T = 10
    dummy = {
        'questions': tf.zeros((1, T), dtype='int32'),
        'inter_questions': tf.zeros((1, T-1), dtype='int32'),
        'inter_responses': tf.zeros((1, T-1), dtype='int32'),
        'inter_features': tf.zeros((1, T-1, n_features), dtype='float32'),
    }
    model(dummy, training=False)
    if rasch_logits is not None:
        model.rasch_difficulty.set_weights([rasch_logits.reshape(-1, 1)])
    return model


# ═══════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOPS
# ═══════════════════════════════════════════════════════════════════════

def skill_loss(y_true, logits, tgt_skills, mask, n_skills, label_smoothing=0.0):
    oh = tf.one_hot(tgt_skills, n_skills)
    pred = tf.reduce_sum(logits * oh, axis=-1)
    if label_smoothing > 0:
        y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=pred)
    return tf.reduce_sum(bce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def akt_loss(y_true, logits, mask, label_smoothing=0.0):
    if label_smoothing > 0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
    return tf.reduce_sum(bce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def train_saint(model, Xtr, ytr, mtr, ttr, Xte, yte, mte, tte,
                n_skills, n_epochs=30, lr=0.001, batch_size=16,
                warmup_epochs=3, patience=15, label='',
                weight_decay=0.0, grad_clip=1.0):
    opt = keras.optimizers.Adam(lr)
    best_auc, best_w, pat_ctr = 0, None, 0
    n_batches = (len(ytr) + batch_size - 1) // batch_size

    wd_vars = ([v for v in model.trainable_variables
                if 'bias' not in v.name and 'layer_normalization' not in v.name]
               if weight_decay > 0 else [])

    for epoch in range(n_epochs):
        if warmup_epochs > 0:
            if epoch < warmup_epochs:
                opt.learning_rate.assign(lr * (epoch + 1) / warmup_epochs)
            elif epoch == warmup_epochs:
                opt.learning_rate.assign(lr)

        perm = np.random.permutation(len(ytr))
        epoch_loss = 0.0
        for start in range(0, len(ytr), batch_size):
            idx = perm[start:start + batch_size]
            bi = {k: tf.constant(v[idx]) for k, v in Xtr.items()}
            by, bm, bt = tf.constant(ytr[idx]), tf.constant(mtr[idx]), tf.constant(ttr[idx])
            with tf.GradientTape() as tape:
                logits = model(bi, training=True)
                loss = skill_loss(by, logits, bt, bm, n_skills)
                if weight_decay > 0:
                    l2 = tf.add_n([tf.reduce_sum(tf.square(v)) for v in wd_vars])
                    loss = loss + weight_decay * l2
            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [(tf.clip_by_norm(g, grad_clip), v)
                              for g, v in zip(grads, model.trainable_variables) if g is not None]
            opt.apply_gradients(grads_and_vars)
            epoch_loss += float(loss)

        logits = model.predict(Xte, batch_size=64, verbose=0)
        oh = tf.one_hot(tte, n_skills).numpy()
        probs = 1 / (1 + np.exp(-np.sum(logits * oh, axis=-1)))
        fp, ft = probs[mte > 0], yte[mte > 0]
        val_auc = roc_auc_score(ft, fp) if len(np.unique(ft)) > 1 else 0.5
        val_bce = -np.mean(ft * np.log(np.clip(fp, 1e-7, 1)) +
                           (1 - ft) * np.log(np.clip(1 - fp, 1e-7, 1)))

        marker = '*' if val_auc > best_auc else ''
        print(f'      {label}ep{epoch+1:02d} loss={epoch_loss/n_batches:.4f} '
              f'val_loss={val_bce:.4f} auc={val_auc:.4f}{marker}', flush=True)

        if val_auc > best_auc:
            best_auc, best_w, pat_ctr = val_auc, model.get_weights(), 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f'      early stop (patience={patience})', flush=True)
                break
    model.set_weights(best_w)
    return model, best_auc


def train_akt(model, Xtr, ytr, mtr, Xte, yte, mte,
              n_epochs=30, lr=0.001, batch_size=16,
              warmup_epochs=3, patience=15, label='',
              weight_decay=0.0, grad_clip=1.0):
    opt = keras.optimizers.Adam(lr)
    best_auc, best_w, pat_ctr = 0, None, 0
    n_batches = (len(ytr) + batch_size - 1) // batch_size

    wd_vars = ([v for v in model.trainable_variables
                if 'bias' not in v.name and 'layer_normalization' not in v.name]
               if weight_decay > 0 else [])

    for epoch in range(n_epochs):
        if warmup_epochs > 0:
            if epoch < warmup_epochs:
                opt.learning_rate.assign(lr * (epoch + 1) / warmup_epochs)
            elif epoch == warmup_epochs:
                opt.learning_rate.assign(lr)

        perm = np.random.permutation(len(ytr))
        epoch_loss = 0.0
        for start in range(0, len(ytr), batch_size):
            idx = perm[start:start + batch_size]
            bi = {k: tf.constant(v[idx]) for k, v in Xtr.items()}
            by, bm = tf.constant(ytr[idx]), tf.constant(mtr[idx])
            with tf.GradientTape() as tape:
                logits = model(bi, training=True)
                loss = akt_loss(by, logits, bm)
                if weight_decay > 0:
                    l2 = tf.add_n([tf.reduce_sum(tf.square(v)) for v in wd_vars])
                    loss = loss + weight_decay * l2
            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [(tf.clip_by_norm(g, grad_clip), v)
                              for g, v in zip(grads, model.trainable_variables) if g is not None]
            opt.apply_gradients(grads_and_vars)
            epoch_loss += float(loss)

        logits = model.predict(Xte, batch_size=64, verbose=0)
        probs = 1 / (1 + np.exp(-logits))
        fp, ft = probs[mte > 0], yte[mte > 0]
        val_auc = roc_auc_score(ft, fp) if len(np.unique(ft)) > 1 else 0.5
        val_bce = -np.mean(ft * np.log(np.clip(fp, 1e-7, 1)) +
                           (1 - ft) * np.log(np.clip(1 - fp, 1e-7, 1)))

        marker = '*' if val_auc > best_auc else ''
        print(f'      {label}ep{epoch+1:02d} loss={epoch_loss/n_batches:.4f} '
              f'val_loss={val_bce:.4f} auc={val_auc:.4f}{marker}', flush=True)

        if val_auc > best_auc:
            best_auc, best_w, pat_ctr = val_auc, model.get_weights(), 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f'      early stop (patience={patience})', flush=True)
                break
    model.set_weights(best_w)
    return model, best_auc


# ═══════════════════════════════════════════════════════════════════════
# 7. CONFIGS
# ═══════════════════════════════════════════════════════════════════════

SAVE_DIR = 'models'
os.makedirs(SAVE_DIR, exist_ok=True)

configs = []

# ── SAINT-Lite subject-level (v3: KerpleLog + EAP + Beta smoothing + crop) ──
configs.append({
    'model': 'saint', 'skill_col': 'subject_idx', 'n_skills': N_SKILLS, 'level': 'subject',
    'embed_dim': 56, 'n_heads': 4, 'n_layers': 2, 'ff_dim': 128,
    'dropout': 0.2, 'lr': 1e-3, 'batch_size': 8, 'seq_len': 200,
    'warmup_epochs': 3, 'weight_decay': 0.0, 'grad_clip': 1.0,
    'n_epochs': 300, 'patience': 10, 'crop_prob': 0.5,
    'tag': 'saint_subject_969_v3',
})

# ── SAINT-Lite topic-level (v3) ──
configs.append({
    'model': 'saint', 'skill_col': 'topic_idx', 'n_skills': N_TOPICS, 'level': 'topic',
    'embed_dim': 56, 'n_heads': 4, 'n_layers': 2, 'ff_dim': 128,
    'dropout': 0.2, 'lr': 1e-3, 'batch_size': 8, 'seq_len': 200,
    'warmup_epochs': 3, 'weight_decay': 0.0, 'grad_clip': 1.0,
    'n_epochs': 300, 'patience': 10, 'crop_prob': 0.5,
    'tag': 'saint_topic_969_v3',
})

# ── AKT question-level (proven architecture) ──
configs.append({
    'model': 'akt', 'level': 'question',
    'embed_dim': 64, 'n_heads': 4, 'n_layers': 2, 'ff_dim': 128,
    'dropout': 0.3, 'lr': 1e-3, 'batch_size': 16, 'seq_len': 200,
    'warmup_epochs': 3, 'weight_decay': 0.0, 'grad_clip': 1.0,
    'n_epochs': 150, 'patience': 15,
    'tag': 'akt_question_969',
})

# ── SAINT-Lite question-level ──
configs.append({
    'model': 'saint', 'skill_col': 'question_idx', 'n_skills': N_QUESTIONS, 'level': 'question',
    'embed_dim': 64, 'n_heads': 4, 'n_layers': 2, 'ff_dim': 128,
    'dropout': 0.3, 'lr': 1e-3, 'batch_size': 8, 'seq_len': 200,
    'warmup_epochs': 3, 'weight_decay': 0.0, 'grad_clip': 1.0,
    'n_epochs': 200, 'patience': 20,
    'tag': 'saint_question_969',
})


# ═══════════════════════════════════════════════════════════════════════
# 8. RUN
# ═══════════════════════════════════════════════════════════════════════

print(f'\n{"="*80}')
print(f'TRAINING ON 969 STUDENTS: SAINT-Lite + AKT')
print(f'{"="*80}')
print(f'Students: {df.user_id.nunique()} ({len(tr_users)} train, {len(te_users)} test)')
print(f'Questions: {N_QUESTIONS}, Subjects: {N_SKILLS}, Topics: {N_TOPICS}')
print(f'Features ({N_F}): {FEATURE_COLS}')
print(f'Configs: {len(configs)}')
for i, c in enumerate(configs):
    print(f'  {i+1}) {c["tag"]} — {c["model"]}, {c["level"]}-level, '
          f'{c["n_epochs"]}ep, lr={c["lr"]}')
print(f'{"="*80}', flush=True)

seq_cache = {}
all_results = {}

for ci, hp in enumerate(configs):
    tag = hp['tag']
    model_type = hp['model']

    # Skip configs that already have saved weights
    save_path = os.path.join(SAVE_DIR, f'{tag}_fold0.weights.h5')
    if os.path.exists(save_path):
        print(f'\n  SKIP [{ci+1}/{len(configs)}] {tag} — weights already exist at {save_path}')
        continue

    t0 = time.time()

    tr_users_fold, te_users_fold = folds[0]
    ft_data = df[df['user_idx'].isin(tr_users_fold)]
    train_cont_mean = ft_data[FEATURE_COLS].mean()
    train_cont_std = ft_data[FEATURE_COLS].std().replace(0, 1)
    EVAL_SEQ_LEN = 200

    if model_type == 'saint':
        n_skills = hp['n_skills']
        skill_col = hp['skill_col']

        # Build model
        model = build_saint_lite(
            n_skills=n_skills, n_patterns=N_PATTERNS, n_features=N_F,
            embed_dim=hp['embed_dim'], n_heads=hp['n_heads'],
            n_layers=hp['n_layers'], ff_dim=hp['ff_dim'], dropout=hp['dropout'])
        n_params = sum(np.prod(v.shape) for v in model.trainable_variables)

        # Build sequences
        ft_uids = list(set(ft_data['user_idx'].unique()))
        Xtr, ytr, mtr, ttr = build_saint_sequences(
            ft_data, ft_uids, FEATURE_COLS, skill_col,
            crop_prob=hp.get('crop_prob', 0.0),
            max_seq_len=hp['seq_len'],
            cont_mean=train_cont_mean, cont_std=train_cont_std)

        te_key = f'saint_te_{skill_col}'
        if te_key not in seq_cache:
            seq_cache[te_key] = build_saint_sequences(
                df, list(te_users_fold), FEATURE_COLS, skill_col,
                max_seq_len=EVAL_SEQ_LEN,
                cont_mean=train_cont_mean, cont_std=train_cont_std)
        Xte, yte, mte, tte = seq_cache[te_key]

        print(f'\n{"="*80}', flush=True)
        print(f'Config [{ci+1}/{len(configs)}] {tag}  ({hp["level"]}-level, {n_skills} skills)', flush=True)
        print(f'  SAINT-Lite(d={hp["embed_dim"]}, L={hp["n_layers"]}, H={hp["n_heads"]}, ff={hp["ff_dim"]})'
              f'  params={n_params:,}', flush=True)
        print(f'  lr={hp["lr"]}  bs={hp["batch_size"]}  epochs={hp["n_epochs"]}  patience={hp["patience"]}', flush=True)
        print(f'  [train on {len(ytr)} seqs, test on {len(yte)} seqs]', flush=True)
        print(f'{"="*80}', flush=True)

        model, best_auc = train_saint(
            model, Xtr, ytr, mtr, ttr, Xte, yte, mte, tte,
            n_skills=n_skills,
            n_epochs=hp['n_epochs'], lr=hp['lr'], batch_size=hp['batch_size'],
            patience=hp['patience'], warmup_epochs=hp['warmup_epochs'],
            weight_decay=hp['weight_decay'], grad_clip=hp['grad_clip'])

        # Eval
        logits = model.predict(Xte, batch_size=64, verbose=0)
        oh = tf.one_hot(tte, n_skills).numpy()
        probs = 1 / (1 + np.exp(-np.sum(logits * oh, axis=-1)))
        m = evaluate(yte[mte > 0], probs[mte > 0])

    elif model_type == 'akt':
        model = build_akt(
            N_QUESTIONS, N_SKILLS, N_F,
            embed_dim=hp['embed_dim'], n_heads=hp['n_heads'],
            n_layers=hp['n_layers'], ff_dim=hp['ff_dim'],
            dropout=hp['dropout'], rasch_logits=_rasch_logits)
        n_params = model.count_params()

        ft_uids = list(set(ft_data['user_idx'].unique()))
        Xtr, ytr, mtr = build_akt_sequences(
            ft_data, ft_uids, FEATURE_COLS,
            crop_prob=hp.get('crop_prob', 0.0),
            max_seq_len=hp['seq_len'],
            cont_mean=train_cont_mean, cont_std=train_cont_std)

        te_key = 'akt_te'
        if te_key not in seq_cache:
            seq_cache[te_key] = build_akt_sequences(
                df, list(te_users_fold), FEATURE_COLS,
                max_seq_len=EVAL_SEQ_LEN,
                cont_mean=train_cont_mean, cont_std=train_cont_std)
        Xte, yte, mte = seq_cache[te_key]

        print(f'\n{"="*80}', flush=True)
        print(f'Config [{ci+1}/{len(configs)}] {tag}  (question-level, {N_QUESTIONS} questions)', flush=True)
        print(f'  AKT(d={hp["embed_dim"]}, L={hp["n_layers"]}, H={hp["n_heads"]}, ff={hp["ff_dim"]})'
              f'  params={n_params:,}', flush=True)
        print(f'  lr={hp["lr"]}  bs={hp["batch_size"]}  epochs={hp["n_epochs"]}  patience={hp["patience"]}', flush=True)
        print(f'  [train on {len(ytr)} seqs, test on {len(yte)} seqs]', flush=True)
        print(f'{"="*80}', flush=True)

        model, best_auc = train_akt(
            model, Xtr, ytr, mtr, Xte, yte, mte,
            n_epochs=hp['n_epochs'], lr=hp['lr'], batch_size=hp['batch_size'],
            patience=hp['patience'], warmup_epochs=hp['warmup_epochs'],
            weight_decay=hp['weight_decay'], grad_clip=hp['grad_clip'])

        logits = model.predict(Xte, batch_size=64, verbose=0)
        probs = 1 / (1 + np.exp(-logits))
        m = evaluate(yte[mte > 0], probs[mte > 0])

    elapsed = time.time() - t0
    save_path = os.path.join(SAVE_DIR, f'{tag}_fold0.weights.h5')
    model.save_weights(save_path)

    print(f'\n  => {tag}: AUC={m["auc"]:.4f}  acc={m["accuracy"]:.4f}  '
          f'logloss={m["log_loss"]:.4f}  params={n_params:,}  [{elapsed:.0f}s]', flush=True)

    all_results[tag] = {'metrics': m, 'config': hp, 'n_params': n_params}
    del model
    gc.collect()

# Final summary
print(f'\n\n{"="*80}')
print(f'FINAL RESULTS — 969 STUDENTS')
print(f'{"="*80}')
print(f'{"Tag":<30} {"Model":<8} {"Level":<10} {"AUC":>8} {"Acc":>8} {"LogLoss":>10} {"Params":>10}')
print(f'{"-"*84}')
ranked = sorted(all_results.items(), key=lambda x: -x[1]['metrics']['auc'])
for tag, info in ranked:
    m = info['metrics']
    c = info['config']
    print(f'{tag:<30} {c["model"]:<8} {c["level"]:<10} {m["auc"]:>8.4f} '
          f'{m["accuracy"]:>8.4f} {m["log_loss"]:>10.4f} {info["n_params"]:>10,}')

print(f'\nReference (286 students):')
print(f'  SAINT-Lite subject AUC=0.8612 (kt_saint_lite.py)')
print(f'  AKT question     AUC=0.8003 (kt_akt.py)')
print(f'  LSTM question    AUC=0.7608 (kt_lstm.py)')
print(f'Done!', flush=True)
