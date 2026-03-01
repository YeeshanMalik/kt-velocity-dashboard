"""
Validate AKT-64 model → Mastery Velocity Pipeline end-to-end.

1. Load AKT-64 small (best model, AUC=0.802)
2. Run inference on test students — get P(correct) per interaction
3. Feed predictions into MasteryVelocityPipeline as kt_prediction
4. Show per-topic mastery, velocity, MVS, and recommendations
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mastery_velocity import MasteryVelocityPipeline, normalize_velocity, compute_breadth

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f'TensorFlow: {tf.__version__}')

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING (same pipeline as kt_akt.py)
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv('data/students_286.csv', parse_dates=['created_at', 'submitted_at'])
print(f'Loaded: {len(df):,} interactions, {df.user_id.nunique()} students')

# Load precomputed train-only IRT (avoids data leak from computing on all students)
q_irt = pd.read_csv('data/question_irt_train.csv')
q_irt_2pl = pd.read_csv('data/question_irt_2pl.csv')
print(f'Loaded train-only IRT: {len(q_irt)} questions, 2PL: {len(q_irt_2pl)} questions')

# FSRS v5 retrievability constant
FSRS_FACTOR = 19 / 81  # = 0.9^(1/-0.5) - 1, ensures R(S,S) = 0.9

def add_irt_features(dataframe, q_irt_table, q_irt_2pl_table):
    from irt_fitting import compute_running_eap
    dataframe = dataframe.merge(q_irt_table, on='question_id', how='left')
    dataframe['difficulty_logit'] = dataframe['difficulty_logit'].fillna(0)
    dataframe['discrimination'] = dataframe['discrimination'].fillna(0)
    # Compute real FSRS retrievability
    acc_clip = dataframe['subject_accuracy_prior'].fillna(0.5).clip(0.05, 0.95)
    S = 1 + 2 * acc_clip  # stability proxy
    t_days = dataframe['subject_elapsed_seconds'].fillna(0).clip(0) / 86400.0
    dataframe['fsrs_retrievability'] = ((1 + FSRS_FACTOR * t_days / S) ** (-0.5)).clip(0.0, 1.0)
    # EAP student ability (replaces logit proxy)
    dataframe = dataframe.sort_values(['user_id', 'created_at']).reset_index(drop=True)
    dataframe['student_ability_logit'] = compute_running_eap(dataframe, q_irt_2pl_table, n_quad=41)
    return dataframe

# Build question encoder
df_all = pd.read_csv('data/interactions_fixed.csv', parse_dates=['created_at'])
all_question_ids = sorted(set(df['question_id'].unique()) | set(df_all['question_id'].unique()))
question_encoder = LabelEncoder()
question_encoder.fit(all_question_ids)
N_QUESTIONS = len(all_question_ids)
df['question_idx'] = question_encoder.transform(df['question_id'])
del df_all

df = add_irt_features(df, q_irt, q_irt_2pl)

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

df['elapsed_log']          = np.log1p(df['elapsed_seconds'].fillna(0).clip(0, 86400*30))
df['subject_elapsed_log']  = np.log1p(df['subject_elapsed_seconds'].fillna(0).clip(0, 86400*30))
df['topic_elapsed_log']    = np.log1p(df['topic_elapsed_seconds'].fillna(0).clip(0, 86400*30))
df['subject_accuracy_prior'] = df['subject_accuracy_prior'].fillna(0.5)
df['topic_accuracy_prior']   = df['topic_accuracy_prior'].fillna(0.5)
df['time_spent_log']         = np.log1p(df['total_time_spent'].fillna(0).clip(0, 3600))
df['subject_attempts_log']   = np.log1p(df['subject_attempts_prior'].fillna(0))
df['topic_attempts_log']     = np.log1p(df['topic_attempts_prior'].fillna(0))
df['correct'] = df['is_correct'].astype(int)

IRT_FSRS_FEATURES = [
    'elapsed_log', 'subject_elapsed_log', 'topic_elapsed_log',
    'subject_accuracy_prior', 'topic_accuracy_prior',
    'time_spent_log', 'subject_attempts_log', 'topic_attempts_log',
    'difficulty_logit', 'discrimination',
    'fsrs_retrievability', 'student_ability_logit',
]
N_SKILLS = df['subject_idx'].nunique()
FEATURE_COLS = IRT_FSRS_FEATURES
N_F = len(FEATURE_COLS)
COLD_START_MASK = 5

# Same train/test split
unique_users = df.groupby('user_idx').agg(mean_accuracy=('correct', 'mean')).reset_index()
unique_users['accuracy_bin'] = pd.qcut(unique_users['mean_accuracy'], q=4, labels=False,
                                       duplicates='drop')
tr_idx, te_idx = train_test_split(
    unique_users.index, test_size=0.2, random_state=SEED,
    stratify=unique_users['accuracy_bin'])
test_user_idxs = set(unique_users.iloc[te_idx]['user_idx'])
print(f'Test users: {len(test_user_idxs)}')

# Build taxonomy: {topic_id: subject_id}
# Use subject as topic level (100% coverage) since L2 is only 39% populated
subjects = sorted(df['subject'].unique())
taxonomy = {s: s for s in subjects}  # subject = topic level
user_idx_to_id = dict(zip(df['user_idx'], df['user_id']))
print(f'Taxonomy: {len(taxonomy)} subjects (using subject as topic level, L2 only 39% populated)')


# ═══════════════════════════════════════════════════════════════════════
# 2. AKT MODEL (same architecture as kt_akt.py)
# ═══════════════════════════════════════════════════════════════════════

class AKTModel(keras.Model):
    def __init__(self, n_questions, n_skills, n_features,
                 embed_dim=128, n_heads=8, n_layers=4, ff_dim=256, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.n_questions = n_questions
        self.n_skills = n_skills
        self.embed_dim = embed_dim
        self._n_layers = n_layers
        self.question_emb = layers.Embedding(n_questions + 1, embed_dim, mask_zero=False)
        self.diff_emb = layers.Embedding(n_questions + 1, embed_dim // 4, mask_zero=False)
        self.feature_proj = layers.Dense(embed_dim // 2)
        self.response_emb = layers.Embedding(2, embed_dim // 4)
        self.inter_proj = layers.Dense(embed_dim)
        self.pos_emb_q = layers.Embedding(512, embed_dim)
        self.pos_emb_s = layers.Embedding(512, embed_dim)
        self.time_proj = layers.Dense(embed_dim, activation='tanh')
        self.input_drop = layers.Dropout(dropout)

        self.q_self_attn, self.q_ffn, self.q_ln1, self.q_ln2, self.q_drop, self.q_gammas = [], [], [], [], [], []
        self.s_self_attn, self.s_ffn, self.s_ln1, self.s_ln2, self.s_drop, self.s_gammas = [], [], [], [], [], []
        self.kr_cross_attn, self.kr_ffn, self.kr_ln1, self.kr_ln2, self.kr_drop, self.kr_gammas = [], [], [], [], [], []

        for prefix, attn_list, ffn_list, ln1_list, ln2_list, drop_list, gamma_list in [
            ('q', self.q_self_attn, self.q_ffn, self.q_ln1, self.q_ln2, self.q_drop, self.q_gammas),
            ('s', self.s_self_attn, self.s_ffn, self.s_ln1, self.s_ln2, self.s_drop, self.s_gammas),
            ('kr', self.kr_cross_attn, self.kr_ffn, self.kr_ln1, self.kr_ln2, self.kr_drop, self.kr_gammas),
        ]:
            for i in range(n_layers):
                attn_list.append(layers.MultiHeadAttention(n_heads, embed_dim // n_heads, dropout=dropout))
                ffn_list.append(keras.Sequential([
                    layers.Dense(ff_dim, activation='gelu'), layers.Dropout(dropout),
                    layers.Dense(embed_dim), layers.Dropout(dropout)]))
                ln1_list.append(layers.LayerNormalization(epsilon=1e-6))
                ln2_list.append(layers.LayerNormalization(epsilon=1e-6))
                drop_list.append(layers.Dropout(dropout))
                gamma_list.append(tf.Variable(tf.constant(-1.0, shape=(n_heads, 1, 1)),
                                              trainable=True, name=f'{prefix}_gamma_{i}'))

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
        q_pos = self.pos_emb_q(tf.range(q_len))[None, :, :]
        q_repr = self.input_drop(q_emb + q_pos, training=training)

        iq_emb = self.question_emb(inter_q)
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
              embed_dim=64, n_heads=4, n_layers=2, ff_dim=128, dropout=0.3):
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
    return model


print('\nBuilding AKT-64 small...')
model = build_akt(N_QUESTIONS, N_SKILLS, N_F)
model.load_weights('models/akt_64_small_fold0.weights.h5')
print(f'Loaded: {model.count_params():,} params')


# ═══════════════════════════════════════════════════════════════════════
# 3. RUN AKT INFERENCE ON TEST STUDENTS
# ═══════════════════════════════════════════════════════════════════════

cont_mean = df[FEATURE_COLS].mean()
cont_std  = df[FEATURE_COLS].std().replace(0, 1)

def get_student_predictions(user_idx):
    """Run AKT on one student, return per-position P(correct) aligned with interactions."""
    sort_col = 'submitted_at' if 'submitted_at' in df.columns else 'created_at'
    ud = df[df['user_idx'] == user_idx].sort_values(sort_col)
    if len(ud) < 2:
        return None, None

    qi = ud['question_idx'].values
    co = ud['correct'].values
    ff = (ud[FEATURE_COLS].fillna(0).values - cont_mean.values) / cont_std.values

    max_seq = 100
    T = min(len(qi), max_seq)

    # Pad sequences
    q_pad  = np.zeros((1, T), dtype='int32')
    qi_pad = np.zeros((1, T-1), dtype='int32')
    r_pad  = np.zeros((1, T-1), dtype='int32')
    f_pad  = np.zeros((1, T-1, N_F), dtype='float32')

    q_pad[0, :T]     = qi[-T:]
    qi_pad[0, :T-1]  = qi[-T:-1]
    r_pad[0, :T-1]   = co[-T:-1]
    f_pad[0, :T-1]   = ff[-T:-1]

    inputs = {
        'questions': q_pad,
        'inter_questions': qi_pad,
        'inter_responses': r_pad,
        'inter_features': f_pad,
    }

    logits = model(inputs, training=False).numpy()  # (1, T-1)
    probs = 1 / (1 + np.exp(-logits[0]))  # (T-1,)

    # probs[t] = P(correct at position t+1 given history 0..t)
    # Align: prediction[i] corresponds to interaction at position i+1 in the last-T slice
    # Return the last-T rows of the dataframe + predictions
    ud_slice = ud.iloc[-T:]
    # predictions are for positions 1..T-1 (0-indexed), so interactions 1..T-1
    return ud_slice.iloc[1:].reset_index(drop=True), probs


# ═══════════════════════════════════════════════════════════════════════
# 4. FULL PIPELINE: AKT → MasteryVelocity
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '='*80)
print('END-TO-END VALIDATION: AKT-64 → Mastery Velocity Pipeline')
print('='*80)

# Pick 10 test students with diverse accuracy
test_df = df[df['user_idx'].isin(test_user_idxs)]
user_stats = test_df.groupby('user_idx').agg(
    n=('correct', 'size'), acc=('correct', 'mean')).sort_values('n', ascending=False)
# Pick 10 with enough interactions, spread across accuracy
user_stats = user_stats[user_stats['n'] >= 50]
n_pick = min(10, len(user_stats))
step = max(1, len(user_stats) // n_pick)
sample_users = list(user_stats.iloc[::step].index[:n_pick])

for uid in sample_users:
    real_uid = user_idx_to_id[uid]
    interactions_df, predictions = get_student_predictions(uid)
    if interactions_df is None:
        continue

    # Initialize pipeline for this student
    pipeline = MasteryVelocityPipeline(taxonomy)

    # Get the first timestamp as reference
    ts0 = interactions_df['submitted_at'].iloc[0].timestamp() / 86400.0

    # Process each interaction through the pipeline
    for i in range(len(interactions_df)):
        row = interactions_df.iloc[i]
        topic = row['subject']
        is_correct = bool(row['correct'])
        ts_days = row['submitted_at'].timestamp() / 86400.0 - ts0
        kt_pred = float(predictions[i])

        pipeline.process_interaction(
            student_id=real_uid,
            topic_id=topic,
            is_correct=is_correct,
            timestamp_days=ts_days,
            kt_prediction=kt_pred,
        )

    # Get final MVS
    current_ts = interactions_df['submitted_at'].iloc[-1].timestamp() / 86400.0 - ts0
    mvs_result = pipeline.get_mvs(real_uid, current_ts)

    # Get recommendations
    recs = pipeline.get_recommendations(real_uid, current_ts, top_n=5)

    # Get actual per-topic accuracy for comparison
    user_all = df[df['user_idx'] == uid]
    actual_topic_acc = user_all.groupby('subject')['is_correct'].agg(['mean', 'count'])

    overall_acc = user_all['is_correct'].mean()
    n_total = len(user_all)

    print(f'\n{"="*80}')
    print(f'Student: {str(real_uid)[:20]}  (n={n_total}, acc={overall_acc:.2f})')
    print(f'{"="*80}')

    # MVS breakdown
    print(f'\n  MVS Score:     {mvs_result["mvs"]:.1f}/100')
    print(f'  Mastery Level: {mvs_result["mastery_level"]:.3f}')
    print(f'  Velocity:      {mvs_result["velocity_raw"]:+.4f} (norm={mvs_result["velocity_normalized"]:.3f})')
    print(f'  Consistency:   {mvs_result["consistency"]:.3f}')
    print(f'  Breadth:       {mvs_result["breadth"]:.3f}')

    # Per-topic mastery vs actual
    topic_ms = mvs_result['topic_masteries']
    studied_topics = [(t, m) for t, m in topic_ms.items()
                      if t in actual_topic_acc.index and actual_topic_acc.loc[t, 'count'] >= 3]
    studied_topics.sort(key=lambda x: -x[1])

    if studied_topics:
        print(f'\n  {"Subject":<40} {"KT Mastery":>10} {"Actual Acc":>10} {"N":>5} {"Delta":>8}')
        print(f'  {"-"*75}')
        for topic, mastery in studied_topics:
            actual = actual_topic_acc.loc[topic, 'mean']
            n = int(actual_topic_acc.loc[topic, 'count'])
            delta = mastery - actual
            print(f'  {topic:<40} {mastery:>10.3f} {actual:>10.3f} {n:>5} {delta:>+8.3f}')

    # Recommendations
    if recs:
        print(f'\n  Top 5 Recommended Subjects:')
        for i, rec in enumerate(recs[:5]):
            print(f'    {i+1}. {rec.topic_id:<40} score={rec.total_score:.3f}  '
                  f'decay={rec.decay_urgency:.2f}  gap={rec.coverage_gap:.2f}  '
                  f'mastery={rec.mastery:.2f}')

# ═══════════════════════════════════════════════════════════════════════
# 5. AGGREGATE: Predicted mastery vs actual accuracy correlation
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '='*80)
print('AGGREGATE VALIDATION: KT mastery vs actual accuracy across all test students')
print('='*80)

all_pred, all_actual = [], []

for uid in test_user_idxs:
    interactions_df, predictions = get_student_predictions(uid)
    if interactions_df is None:
        continue

    pipeline = MasteryVelocityPipeline(taxonomy)
    ts0 = interactions_df['submitted_at'].iloc[0].timestamp() / 86400.0

    for i in range(len(interactions_df)):
        row = interactions_df.iloc[i]
        ts_days = row['submitted_at'].timestamp() / 86400.0 - ts0
        pipeline.process_interaction(
            student_id=str(uid), topic_id=row['subject'],
            is_correct=bool(row['correct']),
            timestamp_days=ts_days, kt_prediction=float(predictions[i]))

    # Compare final topic masteries vs actual accuracy
    user_all = df[df['user_idx'] == uid]
    actual_topic_acc = user_all.groupby('subject')['is_correct'].agg(['mean', 'count'])
    current_ts = interactions_df['submitted_at'].iloc[-1].timestamp() / 86400.0 - ts0
    topic_ms = pipeline.mastery.get_all_topic_masteries(str(uid), current_ts)

    for topic, mastery in topic_ms.items():
        if topic in actual_topic_acc.index and actual_topic_acc.loc[topic, 'count'] >= 5:
            all_pred.append(mastery)
            all_actual.append(actual_topic_acc.loc[topic, 'mean'])

all_pred = np.array(all_pred)
all_actual = np.array(all_actual)

if len(all_pred) > 5:
    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(all_pred, all_actual)
    sr, sp = spearmanr(all_pred, all_actual)
    mae = np.mean(np.abs(all_pred - all_actual))

    print(f'\n  N pairs (topic-student, >=5 questions): {len(all_pred)}')
    print(f'  Pearson r  = {pr:.4f} (p={pp:.2e})')
    print(f'  Spearman r = {sr:.4f} (p={sp:.2e})')
    print(f'  MAE        = {mae:.4f}')
    print(f'  Mean pred  = {all_pred.mean():.4f}')
    print(f'  Mean actual= {all_actual.mean():.4f}')

print('\nDONE!')
