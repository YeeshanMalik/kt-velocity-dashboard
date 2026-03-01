"""
Fix interactions.csv by pulling per-question submitted_ats from web DB.
SELECT QUERIES ONLY — no modifications to any DB table.
"""
import pandas as pd
import psycopg2
import numpy as np

# ── Load current interactions.csv ──
print("Loading interactions.csv...")
df = pd.read_csv('data/interactions.csv')
print(f"  {len(df)} rows, {df.user_id.nunique()} users, {df.prelims_quiz_question_id.nunique()} quiz_question_ids")

# ── Connect to web DB (SELECT only) ──
print("Connecting to web DB...")
conn = psycopg2.connect(
    host="prayas-db.cbii0i4yge4n.ap-south-1.rds.amazonaws.com",
    database="upsc",
    user="prayas_db_user",
    password="prayas_ai_dev_2025",
    port="5432"
)
conn.set_session(readonly=True)  # enforce read-only
cur = conn.cursor()

# ── Pull submitted_ats for all quiz_question_ids ──
# Query in batches of 5000 to avoid memory issues
all_ids = df['prelims_quiz_question_id'].unique().tolist()
print(f"Pulling submitted_ats for {len(all_ids)} quiz_question_ids in batches...")

results = []
batch_size = 5000
for i in range(0, len(all_ids), batch_size):
    batch = all_ids[i:i+batch_size]
    placeholders = ','.join(['%s'] * len(batch))
    cur.execute(f"""
        SELECT prelims_quiz_question_id, submitted_ats, question_order
        FROM prelims_quiz_questions
        WHERE prelims_quiz_question_id IN ({placeholders})
    """, batch)
    rows = cur.fetchall()
    results.extend(rows)
    print(f"  Batch {i//batch_size + 1}: fetched {len(rows)} rows (total: {len(results)})")

cur.close()
conn.close()
print(f"Total fetched from DB: {len(results)}")

# ── Build lookup: quiz_question_id -> submitted_at ──
submitted_at_map = {}
question_order_map = {}
for row in results:
    pqq_id = str(row[0])
    submitted_ats = row[1]  # array
    question_order = row[2]
    # submitted_ats[1] is when the student submitted their answer
    if submitted_ats and len(submitted_ats) > 1:
        submitted_at_map[pqq_id] = submitted_ats[1]
    elif submitted_ats and len(submitted_ats) == 1:
        submitted_at_map[pqq_id] = submitted_ats[0]
    if question_order is not None:
        question_order_map[pqq_id] = question_order

# ── Merge submitted_at into dataframe ──
df['submitted_at'] = df['prelims_quiz_question_id'].map(submitted_at_map)
df['question_order'] = df['prelims_quiz_question_id'].map(question_order_map)

matched = df['submitted_at'].notna().sum()
print(f"Matched submitted_at: {matched}/{len(df)} ({100*matched/len(df):.1f}%)")

# For rows without submitted_at, fall back to created_at
df['submitted_at'] = df['submitted_at'].fillna(df['created_at'])
df['submitted_at'] = pd.to_datetime(df['submitted_at'], utc=True)

# ── Sort by user + submitted_at (correct chronological order) ──
print("Sorting by user + submitted_at...")
df = df.sort_values(['user_id', 'submitted_at', 'question_order']).reset_index(drop=True)

# ── Recompute all _prior features in correct order ──
print("Recomputing _prior features in correct temporal order...")

# Recompute interaction_idx
df['interaction_idx'] = df.groupby('user_id').cumcount()

# Recompute elapsed_seconds
df['created_at_dt'] = pd.to_datetime(df['submitted_at'], utc=True)
def compute_elapsed(group):
    ts = group['created_at_dt']
    elapsed = ts.diff().dt.total_seconds().fillna(0).clip(lower=0)
    return elapsed

df['elapsed_seconds'] = df.groupby('user_id', group_keys=False).apply(compute_elapsed)

# Recompute subject-level _prior features
print("  Recomputing subject_* features...")
is_correct = df['is_correct'].astype(int)

# subject_attempt_num: 0-based position within user+subject
df['subject_attempt_num'] = df.groupby(['user_id', 'subject']).cumcount()

# subject_correct_prior: cumulative correct BEFORE this row
df['subject_correct_prior'] = df.groupby(['user_id', 'subject'])['is_correct'].transform(
    lambda x: x.astype(int).cumsum().shift(1).fillna(0)
)

# subject_attempts_prior: cumulative attempts BEFORE this row
df['subject_attempts_prior'] = df.groupby(['user_id', 'subject']).cumcount()

# subject_accuracy_prior — Laplace (Beta) smoothing: (correct+1)/(attempts+2), default 0.5
df['subject_accuracy_prior'] = (df['subject_correct_prior'] + 1) / (df['subject_attempts_prior'] + 2)

# subject_elapsed_seconds: time since last same-subject question
def compute_subject_elapsed(group):
    ts = group['created_at_dt']
    elapsed = ts.diff().dt.total_seconds().fillna(0).clip(lower=0)
    return elapsed

df['subject_elapsed_seconds'] = df.groupby(['user_id', 'subject'], group_keys=False).apply(compute_subject_elapsed)

# Recompute topic-level _prior features (using L2 as topic, matching serious_students_fixed.csv)
print("  Recomputing topic_* features...")
# Use subject as topic fallback if L2 is missing
topic_col = 'subject'  # interactions.csv may not have L2 populated
if 'L2' in df.columns and df['L2'].notna().sum() > len(df) * 0.5:
    topic_col = 'L2'
    print(f"    Using L2 as topic column ({df['L2'].notna().sum()}/{len(df)} non-null)")
else:
    print(f"    L2 mostly null, using subject as topic fallback")

df['topic_attempt_num'] = df.groupby(['user_id', topic_col]).cumcount()

df['topic_correct_prior'] = df.groupby(['user_id', topic_col])['is_correct'].transform(
    lambda x: x.astype(int).cumsum().shift(1).fillna(0)
)

df['topic_attempts_prior'] = df.groupby(['user_id', topic_col]).cumcount()

# topic_accuracy_prior — Laplace (Beta) smoothing: (correct+1)/(attempts+2), default 0.5
df['topic_accuracy_prior'] = (df['topic_correct_prior'] + 1) / (df['topic_attempts_prior'] + 2)

def compute_topic_elapsed(group):
    ts = group['created_at_dt']
    elapsed = ts.diff().dt.total_seconds().fillna(0).clip(lower=0)
    return elapsed

df['topic_elapsed_seconds'] = df.groupby(['user_id', topic_col], group_keys=False).apply(compute_topic_elapsed)

# ── Drop temp columns, save ──
df = df.drop(columns=['created_at_dt'])
df.to_csv('data/interactions_fixed.csv', index=False)
print(f"\nSaved data/interactions_fixed.csv: {len(df)} rows")

# ── Verify monotonicity ──
print("\nVerifying subject_attempts_prior monotonicity...")
non_mono = 0
for (uid, subj), grp in df.groupby(['user_id', 'subject']):
    vals = grp['subject_attempts_prior'].values
    if not np.all(vals[1:] >= vals[:-1]):
        non_mono += 1
print(f"  Non-monotonic user+subject combos: {non_mono}")

print("\nDone!")
