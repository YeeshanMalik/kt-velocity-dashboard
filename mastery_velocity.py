"""
Mastery Velocity Framework

Components:
  1. FSRS Memory Model — stability, retrievability, update rules
  2. Mastery Aggregation — 3 levels (question, topic, subject)
  3. Dual-Axis Velocity — V_eff (per interaction), V_pace (per day)
     3b. ZPDES Learning Progress (Clement et al., 2015)
     3c. KT Logit Derivative (zero-cost KT-based velocity)
     3d. CUSUM Changepoint Detection (regime shift detection)
  4. MVS Score — composite mastery velocity score
  5. Recommendation Scoring — decay urgency + coverage + composite

All formulas from PLAN.md Sections 5B, 6, 10, 13, 23.6.
"""
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# 1. FSRS MEMORY MODEL
# ═══════════════════════════════════════════════════════════════════════

# FSRS v5 default parameters (population-level)
# From open-spaced-repetition/py-fsrs v5 defaults
FSRS_DEFAULTS = {
    'w0':  0.40255,   # initial stability for grade=1
    'w1':  1.18385,   # initial stability for grade=2
    'w2':  3.17300,   # initial stability for grade=3
    'w3':  15.69105,  # initial stability for grade=4
    'w4':  7.19490,   # base initial difficulty
    'w5':  0.53450,   # difficulty grade sensitivity
    'w6':  1.46040,   # difficulty update rate
    'w7':  0.00460,   # difficulty mean reversion
    'w8':  1.54575,   # stability growth base (e^w8 ≈ 4.7)
    'w9':  0.11920,   # stability diminishing returns exponent
    'w10': 1.01925,   # desirable difficulty exponent
    'w11': 1.93950,   # lapse stability scaling
    'w12': 0.11000,   # lapse difficulty exponent
    'w13': 0.29605,   # lapse recovery exponent
    'w14': 2.26980,   # lapse R-dependency
    'w15': 0.23150,   # hard penalty (applied to SInc)
    'w16': 2.98980,   # easy bonus (applied to SInc)
    'w17': 0.51655,   # same-day review penalty
    'w18': 0.66210,   # same-day review scaling
}

# FSRS v5 forgetting curve constants
# R(t, S) = (1 + FACTOR * t / S) ^ DECAY
# Constraint: R(S, S) = 0.9 (S = days until R drops to 0.9)
FSRS_DECAY = -0.5
FSRS_FACTOR = 19 / 81  # = 0.9^(1/DECAY) - 1 ≈ 0.2346

# ── Recommendation scoring constants (ITZS: Information-Theoretic ZPD Scoring)
# ZPD peak: Wilson et al. (2019) "The 85% Rule for Optimal Learning"
# Exact: optimal accuracy = 1 - erfc(1/√2)/2 for Gaussian noise gradient descent
ZPD_PEAK = 1.0 - 0.5 * math.erfc(1.0 / math.sqrt(2.0))  # = 0.84134...
ZPD_SIGMA = 0.15           # Gaussian width of ZPD band (hyperparameter)
REVIEW_R_THRESHOLD = 0.70  # sigmoid midpoint: review when R drops below this
REVIEW_SIGMOID_K = 10.0    # sigmoid steepness (controls transition sharpness)
NOVELTY_DECAY_N = 5        # interactions before novelty bonus vanishes

# Default ITZS component weights (sum = 1.0)
ITZS_WEIGHTS = {
    'w_elg':     0.35,  # Expected Learning Gain: zpd(P_kt) × (1 - mastery)
    'w_review':  0.25,  # Review urgency: sigmoid(R_threshold - R)
    'w_info':    0.15,  # Information gain: H(P_kt) × (1 - mastery)
    'w_exam':    0.15,  # Exam importance: pyq_weight × (1 - mastery)
    'w_novelty': 0.10,  # Novelty: zpd(P_kt) × recency_decay
}


@dataclass
class FSRSState:
    """Per-item FSRS state for one student-topic/question pair."""
    stability: float = 0.0     # S: days until R drops to 0.9
    difficulty: float = 5.0    # D: item difficulty [1, 10]
    last_review_ts: float = 0  # timestamp of last review (days)
    n_reviews: int = 0         # total review count
    n_lapses: int = 0          # number of failures


class FSRSModel:
    """FSRS v5 memory model.

    Tracks stability and retrievability per student-item pair.

    Key equations (FSRS v5, py-fsrs):
      Retrievability:   R(t, S) = (1 + FACTOR * t/S)^DECAY   where FACTOR=19/81, DECAY=-0.5
      Stability growth: S' = S * (SInc * hard_or_easy + 1)
                         SInc = e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1)
      Stability lapse:  S' = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R))
      Difficulty update: D' = w7 * D0(4) + (1-w7) * (D + delta_D * (10-D)/9)
      Initial difficulty: D0(G) = w4 - e^(w5*(G-1)) + 1
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        self.w = {**FSRS_DEFAULTS, **(params or {})}

    def initial_stability(self, grade: int) -> float:
        """S0 for first review. grade ∈ {1,2,3,4}."""
        grade = max(1, min(4, grade))
        key = f'w{grade - 1}'
        return max(0.1, self.w[key])

    def initial_difficulty(self, grade: int) -> float:
        """D0 for first review. FSRS v5: D0(G) = w4 - e^(w5*(G-1)) + 1."""
        d0 = self.w['w4'] - np.exp(self.w['w5'] * (grade - 1)) + 1
        return np.clip(d0, 1.0, 10.0)

    def retrievability(self, t_days: float, stability: float) -> float:
        """FSRS v5 power-law forgetting: R(t,S) = (1 + FACTOR*t/S)^DECAY.

        Satisfies R(S, S) = 0.9 exactly.
        """
        if stability <= 0:
            return 0.0
        return (1 + FSRS_FACTOR * t_days / stability) ** FSRS_DECAY

    def update(self, state: FSRSState, grade: int, current_ts: float) -> FSRSState:
        """Update FSRS state after a review.

        Args:
            state: current state
            grade: 1=Again(wrong), 2=Hard, 3=Good(correct), 4=Easy
            current_ts: current timestamp in days

        Returns:
            Updated FSRSState
        """
        w = self.w
        new_state = FSRSState(
            last_review_ts=current_ts,
            n_reviews=state.n_reviews + 1,
            n_lapses=state.n_lapses,
        )

        if state.n_reviews == 0:
            # First review — cold start
            new_state.stability = self.initial_stability(grade)
            new_state.difficulty = self.initial_difficulty(grade)
            if grade == 1:
                new_state.n_lapses = 1
            return new_state

        # Time since last review
        t_days = max(0, current_ts - state.last_review_ts)
        R = self.retrievability(t_days, state.stability)
        S = state.stability
        D = state.difficulty

        # FSRS v5 difficulty update: linear damping + mean reversion toward D0(4)
        delta_D = -w['w6'] * (grade - 3)
        D_after = D + delta_D * (10 - D) / 9  # linear damping prevents extremes
        D0_4 = self.initial_difficulty(4)      # mean reversion target
        new_D = w['w7'] * D0_4 + (1 - w['w7']) * D_after
        new_D = np.clip(new_D, 1.0, 10.0)
        new_state.difficulty = new_D

        if grade >= 2:  # Successful recall (Hard, Good, Easy)
            # Stability Increment (SInc)
            SInc = (np.exp(w['w8'])
                    * (11 - D)
                    * S ** (-w['w9'])
                    * (np.exp(w['w10'] * (1 - R)) - 1))

            # FSRS v5: hard/easy penalty applied to SInc BEFORE adding 1
            if grade == 2:
                SInc *= w['w15']
            elif grade == 4:
                SInc *= w['w16']

            new_S = S * (1 + SInc)
            new_state.stability = max(0.1, new_S)
        else:
            # Failed recall (Again) — lapse
            new_S = (w['w11']
                     * D ** (-w['w12'])
                     * ((S + 1) ** w['w13'] - 1)
                     * np.exp(w['w14'] * (1 - R)))
            new_state.stability = max(0.1, min(new_S, S))  # can't exceed pre-lapse S
            new_state.n_lapses = state.n_lapses + 1

        return new_state


# ═══════════════════════════════════════════════════════════════════════
# 2. MASTERY AGGREGATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TopicMastery:
    """Mastery state for one topic, tracked as a Beta distribution.

    Beta(alpha, beta_param) gives:
      - Point estimate: alpha / (alpha + beta_param)
      - Variance: alpha*beta_param / ((alpha+beta_param)^2 * (alpha+beta_param+1))
      - Confidence: 1 - 12*Var  (0 = uniform prior, 1 = near-certain)
    """
    topic_id: str
    alpha: float = 1.0            # Beta distribution: correct evidence
    beta_param: float = 1.0       # Beta distribution: incorrect evidence
    mastery: float = 0.5          # point estimate = alpha / (alpha + beta_param)
    confidence: float = 0.0       # certainty in estimate [0, 1]
    fsrs_state: FSRSState = field(default_factory=FSRSState)
    n_interactions: int = 0
    n_correct: int = 0
    kt_sourced: bool = False      # True if prior was set by KT model prediction
    history: List[Tuple[float, float]] = field(default_factory=list)
    # history = [(timestamp_days, mastery_at_that_time), ...]


@dataclass
class TopicKalmanState:
    """Per-topic Extended Kalman Filter state for one student.

    State vector x = [theta, alpha]:
      - theta: student ability on the IRT logit scale (sigmoid(theta) = mastery)
      - alpha: learning rate (logit-units per interaction)

    Covariance P (2x2): uncertainty in [theta, alpha].
    """
    topic_id: str
    theta: float = 0.0               # ability estimate (logit scale)
    alpha: float = 0.0               # learning rate (logit-units per interaction)
    P: np.ndarray = field(default_factory=lambda: np.array([[1.0, 0.0], [0.0, 0.01]]))
    mastery: float = 0.5             # sigmoid(theta) — cached
    confidence: float = 0.0          # 1 / (1 + sqrt(P[0,0])) — logistic transform
    fsrs_state: FSRSState = field(default_factory=FSRSState)
    n_interactions: int = 0
    n_correct: int = 0
    last_timestamp: float = 0.0
    kt_sourced: bool = False
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (ts, mastery, confidence)


class MasteryTracker:
    """Beta-Bayesian mastery tracker at 3 levels: question, topic, subject.

    Each topic's mastery is modeled as a Beta(α, β) distribution:
      - Point estimate = α / (α + β)
      - Confidence = 1 - 12·Var(Beta), range [0, 1]

    When a KT prediction is available, it anchors the Beta prior (mean = P_kt,
    strength scaled by interaction count). The observed outcome then performs a
    proper Bayesian update, weighted by question discrimination (IRT).

    Without KT, the Beta accumulates evidence directly with exponential
    forgetting so that recent interactions matter more than old ones.

    Subject mastery = confidence-weighted mean of topic masteries.
    Overall mastery = PYQ-weighted + confidence-weighted average.
    """

    def __init__(self, taxonomy: Dict[str, str],
                 pyq_weights: Optional[Dict[str, float]] = None,
                 fsrs_params: Optional[Dict[str, float]] = None,
                 kt_prior_strength: float = 4.0,
                 beta_decay: float = 0.95):
        """
        Args:
            taxonomy: {topic_id: subject_id} mapping
            pyq_weights: {topic_id: weight} for PYQ frequency weighting
            fsrs_params: custom FSRS parameters (or None for defaults)
            kt_prior_strength: pseudo-observation strength when using KT as
                prior. Higher = more trust in KT, slower individual response.
            beta_decay: per-interaction decay for Beta params when no KT.
                Shrinks toward uniform prior. 0.95 ≈ 20-interaction window.
        """
        self.taxonomy = taxonomy  # topic -> subject
        self.subjects = {}  # subject -> [topic_ids]
        for topic, subject in taxonomy.items():
            self.subjects.setdefault(subject, []).append(topic)

        self.pyq_weights = pyq_weights or {t: 1.0 for t in taxonomy}
        self.fsrs = FSRSModel(fsrs_params)
        self.kt_prior_strength = kt_prior_strength
        self.beta_decay = beta_decay

        # Per-student state: {student_id: {topic_id: TopicMastery}}
        self.students: Dict[str, Dict[str, TopicMastery]] = {}

    def _get_topic_state(self, student_id: str, topic_id: str) -> TopicMastery:
        if student_id not in self.students:
            self.students[student_id] = {}
        if topic_id not in self.students[student_id]:
            self.students[student_id][topic_id] = TopicMastery(topic_id=topic_id)
        return self.students[student_id][topic_id]

    def _peek_topic_state(self, student_id: str, topic_id: str):
        """Read-only lookup — does NOT create state for unseen topics.

        Returns TopicMastery or None.
        """
        if student_id not in self.students:
            return None
        return self.students[student_id].get(topic_id)

    @staticmethod
    def _beta_confidence(alpha: float, beta_param: float) -> float:
        """Confidence from Beta distribution concentration.

        Var(Beta(a,b)) = ab / ((a+b)^2 (a+b+1))
        Max variance (Beta(1,1)) = 1/12.
        Confidence = 1 - 12·Var, clipped to [0, 1].
        """
        ab = alpha + beta_param
        var = alpha * beta_param / (ab * ab * (ab + 1))
        return float(np.clip(1.0 - 12.0 * var, 0.0, 1.0))

    def update(self, student_id: str, topic_id: str,
               is_correct: bool, timestamp_days: float,
               kt_prediction: Optional[float] = None,
               discrimination: float = 0.3,
               difficulty: float = 0.0) -> float:
        """Process one interaction via Beta-Bayesian update.

        Args:
            student_id: student identifier
            topic_id: L2 topic identifier
            is_correct: whether the answer was correct
            timestamp_days: timestamp in days (monotonically increasing)
            kt_prediction: KT model's P(correct) BEFORE this outcome.
                If provided, sets the Beta prior mean = P_kt.
            discrimination: IRT point-biserial r for this question.
                Controls observation weight: higher discrimination means
                more informative question → larger Bayesian update.

        Returns:
            Updated topic mastery [0, 1]
        """
        ts = self._get_topic_state(student_id, topic_id)

        # FSRS grade: correct=3 (Good), incorrect=1 (Again)
        grade = 3 if is_correct else 1
        ts.fsrs_state = self.fsrs.update(ts.fsrs_state, grade, timestamp_days)

        ts.n_interactions += 1
        if is_correct:
            ts.n_correct += 1

        # ── Observation weight from IRT discrimination ──
        # High discrimination → question is informative → larger update
        # Clip to [0.1, 2.0] to prevent degenerate updates
        obs_weight = float(np.clip(discrimination, 0.1, 2.0))

        if kt_prediction is not None:
            # ── KT-anchored Bayesian update ──
            # 1. Set Beta prior from KT prediction:
            #    mean = P_kt, concentration = kt_prior_strength
            kt_p = float(np.clip(kt_prediction, 0.01, 0.99))
            strength = self.kt_prior_strength
            alpha_prior = kt_p * strength
            beta_prior = (1.0 - kt_p) * strength

            # 2. Bayesian update: add obs_weight to the winning side
            if is_correct:
                ts.alpha = alpha_prior + obs_weight
                ts.beta_param = beta_prior
            else:
                ts.alpha = alpha_prior
                ts.beta_param = beta_prior + obs_weight

            ts.kt_sourced = True
        else:
            # ── Pure Beta-Bayesian with exponential forgetting ──
            # Decay toward uniform prior Beta(1,1) before updating,
            # so recent interactions are weighted more than old ones.
            ts.alpha = 1.0 + self.beta_decay * (ts.alpha - 1.0)
            ts.beta_param = 1.0 + self.beta_decay * (ts.beta_param - 1.0)

            if is_correct:
                ts.alpha += obs_weight
            else:
                ts.beta_param += obs_weight

            ts.kt_sourced = False

        # ── Point estimate and confidence ──
        ts.mastery = ts.alpha / (ts.alpha + ts.beta_param)
        ts.confidence = self._beta_confidence(ts.alpha, ts.beta_param)

        ts.history.append((timestamp_days, ts.mastery, ts.confidence))
        return ts.mastery

    def get_topic_mastery(self, student_id: str, topic_id: str,
                          current_ts: Optional[float] = None) -> float:
        """Get current mastery for a topic, decayed by FSRS if time has passed.

        FSRS decay shrinks the Beta distribution toward uniform Beta(1,1),
        which elegantly reduces both the point estimate AND the confidence
        as time passes without practice.
        """
        if student_id not in self.students:
            return 0.5
        if topic_id not in self.students[student_id]:
            return 0.5

        ts = self.students[student_id][topic_id]
        if ts.n_interactions == 0:
            return 0.5

        if current_ts is not None and ts.fsrs_state.stability > 0:
            t_days = max(0, current_ts - ts.fsrs_state.last_review_ts)
            R = self.fsrs.retrievability(t_days, ts.fsrs_state.stability)
            # Decay Beta params toward prior Beta(1,1):
            #   α_decayed = 1 + R·(α - 1),  β_decayed = 1 + R·(β - 1)
            # As R→0, distribution→Beta(1,1)=uniform, mastery→0.5
            # As R→1, distribution unchanged
            a_dec = 1.0 + R * (ts.alpha - 1.0)
            b_dec = 1.0 + R * (ts.beta_param - 1.0)
            return a_dec / (a_dec + b_dec)
        return ts.mastery

    def get_topic_confidence(self, student_id: str, topic_id: str,
                             current_ts: Optional[float] = None) -> float:
        """Get confidence in topic mastery estimate, decayed by FSRS."""
        if student_id not in self.students:
            return 0.0
        if topic_id not in self.students[student_id]:
            return 0.0

        ts = self.students[student_id][topic_id]
        if ts.n_interactions == 0:
            return 0.0

        if current_ts is not None and ts.fsrs_state.stability > 0:
            t_days = max(0, current_ts - ts.fsrs_state.last_review_ts)
            R = self.fsrs.retrievability(t_days, ts.fsrs_state.stability)
            a_dec = 1.0 + R * (ts.alpha - 1.0)
            b_dec = 1.0 + R * (ts.beta_param - 1.0)
            return self._beta_confidence(a_dec, b_dec)
        return ts.confidence

    def get_subject_mastery(self, student_id: str, subject_id: str,
                            current_ts: Optional[float] = None) -> float:
        """Subject mastery = confidence-weighted mean of topic masteries.

        Topics with higher confidence (more data, more recent practice)
        contribute more to the subject-level estimate. Falls back to
        equal weighting if all confidences are near zero.
        """
        topics = self.subjects.get(subject_id, [])
        if not topics:
            return 0.5

        total_w = 0.0
        weighted_m = 0.0
        for t in topics:
            m = self.get_topic_mastery(student_id, t, current_ts)
            c = self.get_topic_confidence(student_id, t, current_ts)
            w = max(c, 0.01)  # floor to avoid zero-weight
            weighted_m += w * m
            total_w += w

        return weighted_m / total_w if total_w > 0 else 0.5

    def get_overall_mastery(self, student_id: str,
                            current_ts: Optional[float] = None) -> float:
        """Overall mastery = PYQ-weighted × confidence-weighted average.

        Only includes topics where the student has at least one interaction,
        avoiding dilution from unseen topics pinned at the 0.5 prior.
        Combined weight = pyq_weight × confidence.
        """
        if student_id not in self.students:
            return 0.5

        total_w = 0.0
        weighted_m = 0.0
        for topic_id in self.taxonomy:
            if topic_id not in self.students[student_id]:
                continue
            ts = self.students[student_id][topic_id]
            if ts.n_interactions == 0:
                continue

            m = self.get_topic_mastery(student_id, topic_id, current_ts)
            c = self.get_topic_confidence(student_id, topic_id, current_ts)
            pyq_w = self.pyq_weights.get(topic_id, 1.0)
            w = pyq_w * max(c, 0.01)
            weighted_m += w * m
            total_w += w

        if total_w == 0:
            return 0.5
        return weighted_m / total_w

    def get_all_topic_masteries(self, student_id: str,
                                current_ts: Optional[float] = None) -> Dict[str, float]:
        """Get mastery for all topics."""
        return {t: self.get_topic_mastery(student_id, t, current_ts)
                for t in self.taxonomy}

    def get_all_topic_confidences(self, student_id: str,
                                  current_ts: Optional[float] = None) -> Dict[str, float]:
        """Get confidence for all topics."""
        return {t: self.get_topic_confidence(student_id, t, current_ts)
                for t in self.taxonomy}


class KalmanMasteryTracker:
    """Extended Kalman Filter mastery tracker at 3 levels: question, topic, subject.

    State vector x = [theta, alpha]:
      - theta: student ability on the IRT logit scale
      - alpha: learning rate (logit-units per interaction)

    State transition (per interaction, with time gap dt):
      theta_t = gamma(dt) * theta_{t-1} + alpha_{t-1} + w_theta
      alpha_t = alpha_{t-1} + w_alpha

    where gamma(dt) = FSRS retrievability (retention factor, decays with time).

    Observation model (2PL IRT):
      P(correct) = sigmoid(a * (theta - b))

    where a = question discrimination, b = question difficulty.

    The EKF linearizes the sigmoid via its Jacobian and performs
    Bayesian updates with each binary response. The Joseph form
    covariance update ensures numerical stability.

    Mastery = sigmoid(theta). Confidence = 1 / (1 + sqrt(P[0,0])).
    Subject mastery = confidence-weighted mean of topic masteries.

    References:
      - Knewton (2015). Student Latent State Estimation with the Kalman Filter.
      - EDM (2024). Uncertainty-preserving deep KT with state-space models.
      - Bock & Aitkin (1981). 2PL IRT model.
    """

    def __init__(self, taxonomy: Dict[str, str],
                 pyq_weights: Optional[Dict[str, float]] = None,
                 fsrs_params: Optional[Dict[str, float]] = None,
                 q_theta: float = 0.003,
                 q_alpha: float = 0.001,
                 init_theta_var: float = 1.0,
                 init_alpha_var: float = 0.01):
        """
        Args:
            taxonomy: {topic_id: subject_id} mapping
            pyq_weights: {topic_id: weight} for PYQ frequency weighting
            fsrs_params: custom FSRS parameters (or None for defaults)
            q_theta: process noise variance for mastery per interaction.
                Controls how much mastery fluctuates randomly beyond the
                systematic learning rate trend. 0.01 corresponds to ~0.1
                logit-units std per interaction.
            q_alpha: process noise variance for learning rate per interaction.
                Controls how quickly the learning rate can change. 0.001
                corresponds to ~0.03 logit-units std per interaction.
            init_theta_var: initial variance for theta (prior uncertainty).
                1.0 matches the N(0,1) prior.
            init_alpha_var: initial variance for alpha.
                0.01 allows learning rate in approx [-0.2, 0.2] (95% CI).
        """
        self.taxonomy = taxonomy
        self.subjects: Dict[str, List[str]] = {}
        for topic, subject in taxonomy.items():
            self.subjects.setdefault(subject, []).append(topic)

        self.pyq_weights = pyq_weights or {t: 1.0 for t in taxonomy}
        self.fsrs = FSRSModel(fsrs_params)
        self.q_theta = q_theta
        self.q_alpha = q_alpha
        self.init_theta_var = init_theta_var
        self.init_alpha_var = init_alpha_var

        # Per-student state: {student_id: {topic_id: TopicKalmanState}}
        self.students: Dict[str, Dict[str, TopicKalmanState]] = {}

    def _get_topic_state(self, student_id: str, topic_id: str) -> TopicKalmanState:
        if student_id not in self.students:
            self.students[student_id] = {}
        if topic_id not in self.students[student_id]:
            self.students[student_id][topic_id] = TopicKalmanState(
                topic_id=topic_id,
                P=np.array([[self.init_theta_var, 0.0],
                            [0.0, self.init_alpha_var]]))
        return self.students[student_id][topic_id]

    def _peek_topic_state(self, student_id: str, topic_id: str):
        """Read-only lookup — does NOT create state for unseen topics.

        Returns TopicKalmanState or None.
        """
        if student_id not in self.students:
            return None
        return self.students[student_id].get(topic_id)

    def update(self, student_id: str, topic_id: str,
               is_correct: bool, timestamp_days: float,
               kt_prediction: Optional[float] = None,
               discrimination: float = 1.0,
               difficulty: float = 0.0) -> float:
        """Process one interaction via Extended Kalman Filter update.

        Args:
            student_id: student identifier
            topic_id: L2 topic identifier
            is_correct: whether the answer was correct
            timestamp_days: timestamp in days (monotonically increasing)
            kt_prediction: KT model's P(correct) — used to initialize
                theta on the first interaction for this topic (cold start).
                Ignored on subsequent interactions to avoid double-counting.
            discrimination: 2PL IRT discrimination parameter (a).
                Higher = more informative question. Range [0.1, 5.0].
            difficulty: 2PL IRT difficulty parameter (b) on logit scale.
                0.0 = medium difficulty. Positive = harder.

        Returns:
            Updated topic mastery [0, 1]
        """
        ts = self._get_topic_state(student_id, topic_id)

        # Save old stability BEFORE FSRS update — the EKF prediction step
        # needs the pre-update stability to compute gamma (retention factor).
        old_stability = ts.fsrs_state.stability

        # FSRS grade: correct=3 (Good), incorrect=1 (Again)
        grade = 3 if is_correct else 1
        ts.fsrs_state = self.fsrs.update(ts.fsrs_state, grade, timestamp_days)

        ts.n_interactions += 1
        if is_correct:
            ts.n_correct += 1

        y = 1.0 if is_correct else 0.0
        a = float(np.clip(discrimination, 0.1, 5.0))
        b = float(difficulty)

        # ── Cold start: initialize theta from KT prediction ──
        if ts.n_interactions == 1 and kt_prediction is not None:
            kt_p = float(np.clip(kt_prediction, 0.01, 0.99))
            ts.theta = float(np.log(kt_p / (1.0 - kt_p)))
            # Keep initial covariance (prior uncertainty still applies)

        # ── Prediction step ──
        # Time gap since last interaction
        dt = 0.0
        if ts.last_timestamp > 0 and ts.n_interactions > 1:
            dt = max(0.0, timestamp_days - ts.last_timestamp)

        # Retention factor from FSRS power-law forgetting using OLD stability
        # (before this interaction's FSRS update) so gamma reflects the
        # forgetting that occurred BEFORE the student answered.
        # gamma = (1 + FACTOR * dt / S)^DECAY
        # gamma = 1 when dt = 0 (no forgetting)
        # gamma -> 0 as dt -> infinity (total forgetting)
        if dt > 0 and old_stability > 0:
            gamma = float(self.fsrs.retrievability(dt, old_stability))
        else:
            gamma = 1.0

        # State transition matrix:
        # theta_t = gamma * theta_{t-1} + alpha_{t-1}
        # alpha_t = alpha_{t-1}
        F = np.array([[gamma, 1.0],
                       [0.0,   1.0]])

        x_pred = F @ np.array([ts.theta, ts.alpha])

        # Process noise scaled by time gap (more time = more uncertainty)
        # For continuous-time diffusion discretized at interval dt:
        # Q(dt) = Q_base * dt
        dt_scale = max(1.0, dt)
        Q = np.array([[self.q_theta * dt_scale, 0.0],
                       [0.0, self.q_alpha * dt_scale]])

        P_pred = F @ ts.P @ F.T + Q

        # Floor: forgetting (gamma < 1) should never REDUCE P_pred[0,0] below
        # the no-forgetting (gamma=1) baseline.  When gamma² < 1, F@P@F.T can
        # shrink P[0,0] due to cross-term P[0,1] being weighted by gamma.
        # But forgetting adds uncertainty, never removes it.
        # gamma=1 baseline: P[0,0] + 2*P[0,1] + P[1,1] (from F_identity @ P @ F_identity.T)
        P_floor_theta = (ts.P[0, 0] + 2.0 * ts.P[0, 1]
                         + ts.P[1, 1] + self.q_theta * dt_scale)
        P_pred[0, 0] = max(P_pred[0, 0], P_floor_theta)
        P_pred[1, 1] = max(P_pred[1, 1], ts.P[1, 1] + self.q_alpha * dt_scale)

        # ── Update step (EKF with 2PL IRT observation) ──
        # Observation model: P(correct) = sigmoid(a * (theta - b))
        theta_pred = float(x_pred[0])
        z = a * (theta_pred - b)
        p = 1.0 / (1.0 + math.exp(-z))
        p = max(1e-6, min(1.0 - 1e-6, p))

        # Jacobian of observation function h(x) = sigmoid(a*(theta-b))
        # dh/dtheta = a * p * (1 - p)
        # dh/dalpha = 0  (alpha doesn't affect current observation)
        dp_dtheta = a * p * (1.0 - p)
        # Prevent information starvation at extreme theta: when the sigmoid
        # saturates (p→0 or p→1), dp/dtheta→0 and the Kalman update provides
        # no information, causing P to grow unboundedly. This floor ensures
        # every observation provides minimum diagnostic value.
        dp_dtheta = max(dp_dtheta, 0.05)
        H = np.array([dp_dtheta, 0.0])

        # Observation noise: Bernoulli variance Var(y|theta) = p*(1-p)
        R = p * (1.0 - p)

        # Innovation covariance: S = H * P_pred * H^T + R
        S = float(H @ P_pred @ H) + R

        # Kalman gain: K = P_pred * H^T / S
        K = (P_pred @ H) / S

        # Innovation: difference between actual outcome and predicted probability
        innovation = y - p

        # State update: x = x_pred + K * innovation
        x_new = x_pred + K * innovation

        # Covariance update (Joseph form for numerical stability):
        # P = (I - K*H^T) * P_pred * (I - K*H^T)^T + K * R * K^T
        I_KH = np.eye(2) - np.outer(K, H)
        P_new = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * R

        # Enforce symmetry and add regularization
        P_new = 0.5 * (P_new + P_new.T)
        P_new += np.eye(2) * 1e-8

        # ── Store updated state ──
        ts.theta = float(np.clip(x_new[0], -6.0, 6.0))
        ts.alpha = float(np.clip(x_new[1], -0.5, 0.5))
        ts.P = P_new
        ts.last_timestamp = timestamp_days

        # ── Compute outputs ──
        ts.mastery = 1.0 / (1.0 + math.exp(-ts.theta))
        se_theta = float(np.sqrt(max(0.0, ts.P[0, 0])))
        # Logistic transform: never collapses to 0 even when P is large
        # P=0 → 1.0, P=0.04 → 0.83, P=0.25 → 0.67, P=1.0 → 0.50, P=4.0 → 0.33
        ts.confidence = float(1.0 / (1.0 + se_theta))

        ts.kt_sourced = (kt_prediction is not None and ts.n_interactions == 1)
        ts.history.append((timestamp_days, ts.mastery, ts.confidence))
        return ts.mastery

    def get_topic_mastery(self, student_id: str, topic_id: str,
                          current_ts: Optional[float] = None) -> float:
        """Get current mastery for a topic, with FSRS temporal decay.

        Runs a 'virtual' prediction step to account for forgetting since
        the last observation, without permanently altering the state.
        """
        if student_id not in self.students:
            return 0.5
        if topic_id not in self.students[student_id]:
            return 0.5

        ts = self.students[student_id][topic_id]
        if ts.n_interactions == 0:
            return 0.5

        if current_ts is not None and ts.last_timestamp > 0:
            dt = max(0.0, current_ts - ts.last_timestamp)
            if dt > 0 and ts.fsrs_state.stability > 0:
                gamma = float(self.fsrs.retrievability(
                    dt, ts.fsrs_state.stability))
                # Predicted theta after time gap (virtual prediction)
                theta_pred = gamma * ts.theta + ts.alpha
                return 1.0 / (1.0 + math.exp(-theta_pred))
        return ts.mastery

    def get_topic_confidence(self, student_id: str, topic_id: str,
                             current_ts: Optional[float] = None) -> float:
        """Get confidence in topic mastery, decayed over time gaps.

        Runs a virtual prediction step: covariance grows with elapsed time.
        """
        if student_id not in self.students:
            return 0.0
        if topic_id not in self.students[student_id]:
            return 0.0

        ts = self.students[student_id][topic_id]
        if ts.n_interactions == 0:
            return 0.0

        if current_ts is not None and ts.last_timestamp > 0:
            dt = max(0.0, current_ts - ts.last_timestamp)
            if dt > 0 and ts.fsrs_state.stability > 0:
                gamma = float(self.fsrs.retrievability(
                    dt, ts.fsrs_state.stability))
                F = np.array([[gamma, 1.0], [0.0, 1.0]])
                dt_scale = max(1.0, dt)
                Q = np.array([[self.q_theta * dt_scale, 0.0],
                               [0.0, self.q_alpha * dt_scale]])
                P_pred = F @ ts.P @ F.T + Q
                # Floor: gamma=1 (no-forgetting) baseline — forgetting
                # should only add uncertainty, never reduce it
                P_floor_theta = (ts.P[0, 0] + 2.0 * ts.P[0, 1]
                                 + ts.P[1, 1] + self.q_theta * dt_scale)
                P_pred[0, 0] = max(P_pred[0, 0], P_floor_theta)
                se_theta = float(np.sqrt(max(0.0, P_pred[0, 0])))
                # Confidence must never increase with time gap
                return min(ts.confidence, float(1.0 / (1.0 + se_theta)))
        return ts.confidence

    def get_topic_learning_rate(self, student_id: str,
                                topic_id: str) -> float:
        """Get estimated learning rate (alpha) for a topic.

        Positive = improving, negative = declining, near-zero = stagnant.
        Units: logit-units per interaction.
        """
        if student_id not in self.students:
            return 0.0
        if topic_id not in self.students[student_id]:
            return 0.0
        return self.students[student_id][topic_id].alpha

    def get_subject_mastery(self, student_id: str, subject_id: str,
                            current_ts: Optional[float] = None) -> float:
        """Subject mastery = confidence-weighted mean of topic masteries."""
        topics = self.subjects.get(subject_id, [])
        if not topics:
            return 0.5

        total_w = 0.0
        weighted_m = 0.0
        for t in topics:
            m = self.get_topic_mastery(student_id, t, current_ts)
            c = self.get_topic_confidence(student_id, t, current_ts)
            w = max(c, 0.01)
            weighted_m += w * m
            total_w += w

        return weighted_m / total_w if total_w > 0 else 0.5

    def get_overall_mastery(self, student_id: str,
                            current_ts: Optional[float] = None) -> float:
        """Overall mastery = PYQ-weighted x confidence-weighted average.

        Only includes topics with at least one interaction.
        """
        if student_id not in self.students:
            return 0.5

        total_w = 0.0
        weighted_m = 0.0
        for topic_id in self.taxonomy:
            if topic_id not in self.students[student_id]:
                continue
            ts = self.students[student_id][topic_id]
            if ts.n_interactions == 0:
                continue

            m = self.get_topic_mastery(student_id, topic_id, current_ts)
            c = self.get_topic_confidence(student_id, topic_id, current_ts)
            pyq_w = self.pyq_weights.get(topic_id, 1.0)
            w = pyq_w * max(c, 0.01)
            weighted_m += w * m
            total_w += w

        if total_w == 0:
            return 0.5
        return weighted_m / total_w

    def get_all_topic_masteries(self, student_id: str,
                                current_ts: Optional[float] = None) -> Dict[str, float]:
        """Get mastery for all topics."""
        return {t: self.get_topic_mastery(student_id, t, current_ts)
                for t in self.taxonomy}

    def get_all_topic_confidences(self, student_id: str,
                                  current_ts: Optional[float] = None) -> Dict[str, float]:
        """Get confidence for all topics."""
        return {t: self.get_topic_confidence(student_id, t, current_ts)
                for t in self.taxonomy}


# ═══════════════════════════════════════════════════════════════════════
# 3. VELOCITY TRACKERS
# ═══════════════════════════════════════════════════════════════════════


class BaseVelocityTracker(ABC):
    """Common interface for all velocity tracker implementations."""

    @abstractmethod
    def record(self, student_id: str, timestamp_days: float,
               overall_mastery: float,
               subject_id: Optional[str] = None,
               subject_mastery: Optional[float] = None,
               is_correct: Optional[bool] = None,
               kt_prediction: Optional[float] = None) -> Tuple[float, float]:
        """Record an observation and return (velocity, confidence).

        Args:
            student_id: student identifier
            timestamp_days: timestamp in fractional days
            overall_mastery: current overall mastery
            subject_id: subject that was just studied
            subject_mastery: current mastery for that subject
            is_correct: raw binary correctness (needed by ZPDES, CUSUM)
            kt_prediction: KT model's P(correct) (needed by KT approach)

        Returns:
            (velocity, confidence) where velocity is in [-1, +1] and
            confidence is in [0, 1].
        """
        ...

    @abstractmethod
    def get_subject_velocity(self, student_id: str, subject_id: str) -> float:
        """Get current velocity for a specific subject. Range [-1, +1]."""
        ...

    @abstractmethod
    def get_aggregate_velocity(self, student_id: str) -> float:
        """Get aggregate velocity across all active subjects. Range [-1, +1]."""
        ...

    @abstractmethod
    def get_consistency(self, student_id: str, window: int = 10) -> float:
        """Consistency score [0, 1]. Higher = more consistent."""
        ...

    @abstractmethod
    def normalize_for_mvs(self, velocity: float) -> float:
        """Normalize approach-specific velocity to [0, 1] for MVS input."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for comparison reports."""
        ...


# ─── 3a. Baseline: Dual-Axis Velocity (original) ───────────────────────

@dataclass
class SubjectVelocityState:
    """Per-subject velocity tracking for one student."""
    last_mastery: float = 0.5          # last mastery value for this subject
    v_eff_smoothed: float = 0.0        # EMA of per-interaction velocity
    n_updates: int = 0


@dataclass
class VelocityState:
    """Tracks velocity for one student."""
    # Per-interaction tracking
    mastery_history: List[Tuple[float, float]] = field(default_factory=list)
    # [(timestamp_days, overall_mastery), ...]

    # Smoothed velocities (EMA)
    v_eff_smoothed: float = 0.0    # per-interaction velocity (aggregated from subjects)
    v_pace_smoothed: float = 0.0   # per-day velocity

    # Raw velocity history
    v_eff_history: List[float] = field(default_factory=list)
    v_pace_history: List[float] = field(default_factory=list)

    # Smoothed velocity history (for acceleration)
    v_eff_smoothed_history: List[float] = field(default_factory=list)
    v_pace_smoothed_history: List[float] = field(default_factory=list)

    # Per-day tracking
    daily_mastery: Dict[int, float] = field(default_factory=dict)
    # {day_number: mastery_at_end_of_day}

    # Per-subject velocity tracking
    subject_velocities: Dict[str, SubjectVelocityState] = field(default_factory=dict)


class VelocityTracker:
    """Dual-axis velocity: V_eff (per interaction) and V_pace (per day).

    V_eff tracks per-subject mastery deltas and aggregates them, so that
    changes in one subject aren't diluted by 15 unchanged subjects.

    V_pace(d) = M(d) - M(d-1)                          [pace]
    V̄_pace(d) = α₂·V_pace(d) + (1-α₂)·V̄_pace(d-1)   [smoothed]
    """

    def __init__(self, alpha_eff: float = 0.2, alpha_pace: float = 0.3):
        self.alpha_eff = alpha_eff
        self.alpha_pace = alpha_pace
        self.students: Dict[str, VelocityState] = {}

    def _get_state(self, student_id: str) -> VelocityState:
        if student_id not in self.students:
            self.students[student_id] = VelocityState()
        return self.students[student_id]

    def record(self, student_id: str, timestamp_days: float,
               overall_mastery: float,
               subject_id: Optional[str] = None,
               subject_mastery: Optional[float] = None) -> Tuple[float, float]:
        """Record a mastery observation and compute velocities.

        Args:
            student_id: student identifier
            timestamp_days: timestamp in days
            overall_mastery: current overall mastery (for pace velocity)
            subject_id: subject that was just studied (for per-subject velocity)
            subject_mastery: current mastery of that subject (for per-subject velocity)

        Returns:
            (v_eff_smoothed, v_pace_smoothed)
        """
        vs = self._get_state(student_id)

        # --- Per-subject efficiency velocity ---
        if subject_id is not None and subject_mastery is not None:
            if subject_id not in vs.subject_velocities:
                vs.subject_velocities[subject_id] = SubjectVelocityState(
                    last_mastery=subject_mastery)

            sv = vs.subject_velocities[subject_id]
            if sv.n_updates > 0:
                # Delta within this subject (not diluted by other subjects)
                v_subj = subject_mastery - sv.last_mastery
                sv.v_eff_smoothed = (self.alpha_eff * v_subj
                                     + (1 - self.alpha_eff) * sv.v_eff_smoothed)
            sv.last_mastery = subject_mastery
            sv.n_updates += 1

            # Aggregate: mean of per-subject smoothed velocities
            active_svs = [s for s in vs.subject_velocities.values()
                          if s.n_updates >= 2]
            if active_svs:
                v_eff = np.mean([s.v_eff_smoothed for s in active_svs])
            else:
                v_eff = 0.0
        else:
            # Fallback: overall mastery delta with EMA (backward compatible)
            if len(vs.mastery_history) > 0:
                prev_m = vs.mastery_history[-1][1]
                v_eff_raw = overall_mastery - prev_m
            else:
                v_eff_raw = 0.0
            v_eff = (self.alpha_eff * v_eff_raw
                     + (1 - self.alpha_eff) * vs.v_eff_smoothed)

        vs.v_eff_smoothed = v_eff
        vs.v_eff_history.append(v_eff)
        vs.v_eff_smoothed_history.append(vs.v_eff_smoothed)

        # --- Pace velocity (per day) ---
        day = int(timestamp_days)
        vs.daily_mastery[day] = overall_mastery

        prev_day = day - 1
        if prev_day in vs.daily_mastery:
            v_pace = overall_mastery - vs.daily_mastery[prev_day]
        else:
            v_pace = 0.0

        vs.v_pace_smoothed = (self.alpha_pace * v_pace
                              + (1 - self.alpha_pace) * vs.v_pace_smoothed)
        vs.v_pace_history.append(v_pace)
        vs.v_pace_smoothed_history.append(vs.v_pace_smoothed)

        # Record
        vs.mastery_history.append((timestamp_days, overall_mastery))

        return vs.v_eff_smoothed, vs.v_pace_smoothed

    def get_acceleration(self, student_id: str) -> Tuple[float, float]:
        """A_eff and A_pace (second derivative of smoothed velocity).

        Uses smoothed velocity history so acceleration reflects the
        sustained trend change, not just noise between adjacent steps.
        """
        vs = self._get_state(student_id)
        a_eff = 0.0
        a_pace = 0.0
        if len(vs.v_eff_smoothed_history) >= 2:
            a_eff = vs.v_eff_smoothed_history[-1] - vs.v_eff_smoothed_history[-2]
        if len(vs.v_pace_smoothed_history) >= 2:
            a_pace = vs.v_pace_smoothed_history[-1] - vs.v_pace_smoothed_history[-2]
        return a_eff, a_pace

    def get_mastery_ceiling(self, student_id: str,
                            avg_decay_rate: float = 0.05,
                            daily_effort: float = 10.0) -> float:
        """M_∞ = E · V̄_eff / λ_avg.

        Args:
            avg_decay_rate: λ_avg, average daily mastery decay
            daily_effort: E, average interactions per day
        """
        vs = self._get_state(student_id)
        if vs.v_eff_smoothed <= 0 or avg_decay_rate <= 0:
            return 0.0
        ceiling = daily_effort * vs.v_eff_smoothed / avg_decay_rate
        return min(1.0, max(0.0, ceiling))

    def get_consistency(self, student_id: str, window: int = 10) -> float:
        """Consistency = 1 - normalized variance of recent velocities.

        Returns:
            Consistency score [0, 1]. Higher = more consistent.
        """
        vs = self._get_state(student_id)
        if len(vs.v_eff_history) < 2:
            return 0.5

        recent = vs.v_eff_history[-window:]
        var = np.var(recent)
        # Normalize: max_var = 0.01 (if velocity swings ±0.1 per interaction)
        max_var = 0.01
        return max(0.0, 1.0 - var / max_var)


# ─── 3b. ZPDES Learning Progress (Clement et al., 2015) ────────────────


@dataclass
class ZPDESSubjectState:
    """Per-student per-subject state for ZPDES learning progress."""
    outcomes: List[int] = field(default_factory=list)
    velocity_history: List[float] = field(default_factory=list)


class ZPDESVelocityTracker(BaseVelocityTracker):
    """ZPDES Learning Progress (Clement et al., 2015) with adaptive window.

    LP(topic, t) = mean(outcomes[t-d/2 : t]) - mean(outcomes[t-d : t-d/2])
    Operates on raw binary correctness, NOT smoothed mastery.
    Range: [-1, +1], self-calibrating.

    Adaptive window: d = max(min_d, min(n // 2, max_d)) where n = total
    outcomes for that subject. This lets ZPDES see gradual improvements
    that span more than 16 interactions.
    """

    def __init__(self, window_d: int = 16, min_d: int = 16, max_d: int = 64):
        self.default_d = window_d
        self.min_d = min_d
        self.max_d = max_d
        # {student_id: {subject_id: ZPDESSubjectState}}
        self.students: Dict[str, Dict[str, ZPDESSubjectState]] = {}

    @property
    def name(self) -> str:
        return "ZPDES Learning Progress"

    def _get_state(self, student_id: str, subject_id: str) -> ZPDESSubjectState:
        if student_id not in self.students:
            self.students[student_id] = {}
        if subject_id not in self.students[student_id]:
            self.students[student_id][subject_id] = ZPDESSubjectState()
        return self.students[student_id][subject_id]

    def _effective_d(self, n: int) -> int:
        """Adaptive window size: grows with data, capped at max_d."""
        return max(self.min_d, min(n // 2, self.max_d))

    def _compute_lp(self, outcomes: List[int]) -> Tuple[float, float]:
        """Compute Learning Progress and confidence.

        Returns (lp, confidence).
        """
        n = len(outcomes)
        if n < 2:
            return 0.0, 0.0

        d = self._effective_d(n)
        half_d = d // 2

        recent_start = max(0, n - half_d)
        older_end = recent_start
        older_start = max(0, n - d)

        recent = outcomes[recent_start:]
        older = outcomes[older_start:older_end]

        if not recent or not older:
            # LP is undefined without both windows → confidence must be 0
            return 0.0, 0.0

        lp = float(np.mean(recent) - np.mean(older))
        # Ramp confidence based on older window fill (LP needs both windows).
        n_older = len(older)
        confidence = min(1.0, n_older / half_d)
        return lp, confidence

    def record(self, student_id: str, timestamp_days: float,
               overall_mastery: float,
               subject_id: Optional[str] = None,
               subject_mastery: Optional[float] = None,
               is_correct: Optional[bool] = None,
               kt_prediction: Optional[float] = None) -> Tuple[float, float]:
        if subject_id is None or is_correct is None:
            return 0.0, 0.0

        state = self._get_state(student_id, subject_id)
        state.outcomes.append(1 if is_correct else 0)
        lp, confidence = self._compute_lp(state.outcomes)
        state.velocity_history.append(lp)
        return lp, confidence

    def get_subject_velocity(self, student_id: str, subject_id: str) -> float:
        if (student_id not in self.students or
                subject_id not in self.students[student_id]):
            return 0.0
        state = self.students[student_id][subject_id]
        lp, _ = self._compute_lp(state.outcomes)
        return lp

    def get_aggregate_velocity(self, student_id: str) -> float:
        if student_id not in self.students:
            return 0.0
        velocities = []
        for subj_id, state in self.students[student_id].items():
            if len(state.outcomes) >= 2:
                lp, _ = self._compute_lp(state.outcomes)
                velocities.append(lp)
        return float(np.mean(velocities)) if velocities else 0.0

    def get_consistency(self, student_id: str, window: int = 10) -> float:
        if student_id not in self.students:
            return 0.5
        max_var = 0.25  # LP range is [-1,+1], max var for uniform ≈ 0.33
        per_subj = []
        for state in self.students[student_id].values():
            recent = state.velocity_history[-window:]
            if len(recent) >= 2:
                per_subj.append(max(0.0, 1.0 - np.var(recent) / max_var))
        return float(np.mean(per_subj)) if per_subj else 0.5

    def normalize_for_mvs(self, velocity: float) -> float:
        return float(np.clip((velocity + 1.0) / 2.0, 0.0, 1.0))


# ─── 3c. KT Logit Derivative (zero-cost KT-based) ─────────────────────


@dataclass
class KTSubjectState:
    """Per-student per-subject state for KT logit derivative."""
    last_kt_prediction: Optional[float] = None
    v_smoothed: float = 0.0
    velocity_history: List[float] = field(default_factory=list)
    n_updates: int = 0


class KTVelocityTracker(BaseVelocityTracker):
    """KT Probability Delta — EMA-smoothed change in KT model P(correct).

    NOTE: Despite the class name ("Logit"), this tracker operates in
    probability space, NOT logit space. The computation is:
        v(t) = P_kt(t) - P_kt(t-1), EMA-smoothed
    This avoids numerical instability near P=0 or P=1.
    Already difficulty-controlled since KT incorporates IRT features.
    Range: approximately [-0.3, +0.3]. No cold start — works from
    the second prediction.
    """

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        # {student_id: {subject_id: KTSubjectState}}
        self.students: Dict[str, Dict[str, KTSubjectState]] = {}

    @property
    def name(self) -> str:
        return "KT Logit Derivative"

    def _get_state(self, student_id: str, subject_id: str) -> KTSubjectState:
        if student_id not in self.students:
            self.students[student_id] = {}
        if subject_id not in self.students[student_id]:
            self.students[student_id][subject_id] = KTSubjectState()
        return self.students[student_id][subject_id]

    def record(self, student_id: str, timestamp_days: float,
               overall_mastery: float,
               subject_id: Optional[str] = None,
               subject_mastery: Optional[float] = None,
               is_correct: Optional[bool] = None,
               kt_prediction: Optional[float] = None) -> Tuple[float, float]:
        if subject_id is None or kt_prediction is None:
            return 0.0, 0.0

        state = self._get_state(student_id, subject_id)

        if state.last_kt_prediction is not None:
            v_raw = kt_prediction - state.last_kt_prediction
            state.v_smoothed = (self.ema_alpha * v_raw
                                + (1 - self.ema_alpha) * state.v_smoothed)
            state.n_updates += 1

        state.last_kt_prediction = kt_prediction
        state.velocity_history.append(state.v_smoothed)

        confidence = 1.0 if state.n_updates >= 2 else (
            0.5 if state.n_updates == 1 else 0.0)
        return state.v_smoothed, confidence

    def get_subject_velocity(self, student_id: str, subject_id: str) -> float:
        if (student_id not in self.students or
                subject_id not in self.students[student_id]):
            return 0.0
        return self.students[student_id][subject_id].v_smoothed

    def get_aggregate_velocity(self, student_id: str) -> float:
        if student_id not in self.students:
            return 0.0
        velocities = []
        for state in self.students[student_id].values():
            if state.n_updates >= 1:
                velocities.append(state.v_smoothed)
        return float(np.mean(velocities)) if velocities else 0.0

    def get_consistency(self, student_id: str, window: int = 10) -> float:
        if student_id not in self.students:
            return 0.5
        max_var = 0.04  # EMA-smoothed deltas rarely exceed ±0.2
        per_subj = []
        for state in self.students[student_id].values():
            recent = state.velocity_history[-window:]
            if len(recent) >= 2:
                per_subj.append(max(0.0, 1.0 - np.var(recent) / max_var))
        return float(np.mean(per_subj)) if per_subj else 0.5

    def normalize_for_mvs(self, velocity: float) -> float:
        # Empirical range ~[-0.3, +0.3], map to [0, 1] with 0 → 0.5
        # Consistent with Ensemble SCALE_FACTOR of 3.333
        return float(np.clip((velocity + 0.3) / 0.6, 0.0, 1.0))


# ─── 3d. CUSUM Changepoint Detection ───────────────────────────────────


@dataclass
class CUSUMSubjectState:
    """Per-student per-subject state for CUSUM changepoint detection."""
    s_plus: float = 0.0          # positive CUSUM statistic (improvement)
    s_minus: float = 0.0         # negative CUSUM statistic (decline)
    mu_baseline: float = 0.5     # running baseline accuracy
    n_interactions: int = 0
    v_smoothed: float = 0.0      # EMA-smoothed velocity (dampens reset sawtooth)
    changepoints: List[Tuple[int, str]] = field(default_factory=list)
    velocity_history: List[float] = field(default_factory=list)


class CUSUMVelocityTracker(BaseVelocityTracker):
    """CUSUM Changepoint Detection for learning velocity.

    Two-sided CUSUM:
      S_t+ = max(0, S_{t-1}+ + (x_t - mu_0 - k))   # improvement
      S_t- = max(0, S_{t-1}- + (mu_0 - k - x_t))    # decline

    Outputs:
      - Boolean changepoint flags + direction
      - Continuous velocity signal from normalized CUSUM: (S+ - S-) / h
    """

    def __init__(self, k: float = 0.1, h: float = 4.0,
                 baseline_ema_alpha: float = 0.05,
                 velocity_ema_alpha: float = 0.3):
        self.k = k           # sensitivity parameter
        self.h = h           # threshold for changepoint declaration
        self.baseline_alpha = baseline_ema_alpha
        self.velocity_alpha = velocity_ema_alpha  # smooth raw (S+-S-)/h
        # {student_id: {subject_id: CUSUMSubjectState}}
        self.students: Dict[str, Dict[str, CUSUMSubjectState]] = {}

    @property
    def name(self) -> str:
        return "CUSUM Changepoint"

    def _get_state(self, student_id: str, subject_id: str) -> CUSUMSubjectState:
        if student_id not in self.students:
            self.students[student_id] = {}
        if subject_id not in self.students[student_id]:
            self.students[student_id][subject_id] = CUSUMSubjectState()
        return self.students[student_id][subject_id]

    def _update_cusum(self, state: CUSUMSubjectState,
                      x_t: float) -> Tuple[float, Optional[str]]:
        """Update CUSUM statistics and check for changepoint.

        Returns (continuous_velocity, changepoint_direction_or_None).
        """
        mu = state.mu_baseline

        # Update CUSUM statistics
        state.s_plus = max(0.0, state.s_plus + (x_t - mu - self.k))
        state.s_minus = max(0.0, state.s_minus + (mu - self.k - x_t))

        # Check for changepoint
        direction = None
        if state.s_plus > self.h:
            direction = 'improvement'
            state.changepoints.append((state.n_interactions, direction))
            state.s_plus = 0.0   # reset after detection
        elif state.s_minus > self.h:
            direction = 'decline'
            state.changepoints.append((state.n_interactions, direction))
            state.s_minus = 0.0  # reset after detection

        # Update baseline AFTER CUSUM computation (slow EMA)
        # Clamp to [k+0.05, 1-k-0.05] so both S+ and S- can always accumulate.
        # Without this, mu drifting below k makes S- unable to detect declines.
        mu_raw = (self.baseline_alpha * x_t
                  + (1 - self.baseline_alpha) * state.mu_baseline)
        state.mu_baseline = float(np.clip(mu_raw, self.k + 0.05, 1.0 - self.k - 0.05))

        # Continuous velocity signal — clipped to [-1, +1]
        # (S+ can momentarily exceed h by up to max_increment before reset)
        velocity = float(np.clip(
            (state.s_plus - state.s_minus) / self.h, -1.0, 1.0))
        return velocity, direction

    def record(self, student_id: str, timestamp_days: float,
               overall_mastery: float,
               subject_id: Optional[str] = None,
               subject_mastery: Optional[float] = None,
               is_correct: Optional[bool] = None,
               kt_prediction: Optional[float] = None) -> Tuple[float, float]:
        if subject_id is None or is_correct is None:
            return 0.0, 0.0

        state = self._get_state(student_id, subject_id)
        x_t = 1.0 if is_correct else 0.0
        state.n_interactions += 1

        v_raw, _ = self._update_cusum(state, x_t)
        # EMA-smooth the raw CUSUM velocity to dampen sawtooth discontinuities
        # that occur when S+ or S- resets to 0 at changepoint detection
        state.v_smoothed = (self.velocity_alpha * v_raw
                            + (1 - self.velocity_alpha) * state.v_smoothed)
        state.velocity_history.append(state.v_smoothed)

        confidence = min(1.0, state.n_interactions / 8.0)
        return state.v_smoothed, confidence

    def get_changepoints(self, student_id: str,
                         subject_id: str) -> List[Tuple[int, str]]:
        """Return list of (interaction_idx, direction) changepoints."""
        if (student_id not in self.students or
                subject_id not in self.students[student_id]):
            return []
        return self.students[student_id][subject_id].changepoints

    def get_subject_velocity(self, student_id: str, subject_id: str) -> float:
        if (student_id not in self.students or
                subject_id not in self.students[student_id]):
            return 0.0
        return self.students[student_id][subject_id].v_smoothed

    def get_aggregate_velocity(self, student_id: str) -> float:
        if student_id not in self.students:
            return 0.0
        velocities = []
        for state in self.students[student_id].values():
            if state.n_interactions >= 2:
                velocities.append(state.v_smoothed)
        return float(np.mean(velocities)) if velocities else 0.0

    def get_consistency(self, student_id: str, window: int = 10) -> float:
        if student_id not in self.students:
            return 0.5
        max_var = 0.25  # (S+ - S-)/h range is roughly [-1, +1]
        per_subj = []
        for state in self.students[student_id].values():
            recent = state.velocity_history[-window:]
            if len(recent) >= 2:
                per_subj.append(max(0.0, 1.0 - np.var(recent) / max_var))
        return float(np.mean(per_subj)) if per_subj else 0.5

    def normalize_for_mvs(self, velocity: float) -> float:
        return float(np.clip((velocity + 1.0) / 2.0, 0.0, 1.0))


# ── 3e. MASTERY DELTA VELOCITY TRACKER ─────────────────────────────────

@dataclass
class MasteryDeltaSubjectState:
    """Per-student per-subject state for mastery delta velocity."""
    last_mastery: float = 0.5
    v_smoothed: float = 0.0
    n_updates: int = 0
    velocity_history: List[float] = field(default_factory=list)


class MasteryDeltaVelocityTracker(BaseVelocityTracker):
    """EMA-smoothed mastery deltas — bridges Kalman mastery into velocity.

    v(t) = mastery(t) - mastery(t-1), EMA-smoothed per subject.
    Directly uses the EKF's mastery output (sigmoid(theta)), so
    improvements to the Kalman filter directly improve velocity.
    Range: typically [-0.02, +0.02] per interaction.
    """

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        # {student_id: {subject_id: MasteryDeltaSubjectState}}
        self.students: Dict[str, Dict[str, MasteryDeltaSubjectState]] = {}

    @property
    def name(self) -> str:
        return "Mastery Delta"

    def _get_state(self, student_id: str, subject_id: str) -> MasteryDeltaSubjectState:
        if student_id not in self.students:
            self.students[student_id] = {}
        if subject_id not in self.students[student_id]:
            self.students[student_id][subject_id] = MasteryDeltaSubjectState()
        return self.students[student_id][subject_id]

    def record(self, student_id: str, timestamp_days: float,
               overall_mastery: float,
               subject_id: Optional[str] = None,
               subject_mastery: Optional[float] = None,
               is_correct: Optional[bool] = None,
               kt_prediction: Optional[float] = None) -> Tuple[float, float]:
        if subject_id is None or subject_mastery is None:
            return 0.0, 0.0

        state = self._get_state(student_id, subject_id)

        if state.n_updates > 0:
            delta = subject_mastery - state.last_mastery
            state.v_smoothed = (self.ema_alpha * delta
                                + (1 - self.ema_alpha) * state.v_smoothed)

        state.last_mastery = subject_mastery
        state.n_updates += 1
        state.velocity_history.append(state.v_smoothed)

        # Confidence: ramp from 0 → 1 over 8 interactions (needs deltas)
        confidence = min(1.0, max(0.0, (state.n_updates - 1) / 8.0))
        return state.v_smoothed, confidence

    def get_subject_velocity(self, student_id: str, subject_id: str) -> float:
        if (student_id not in self.students or
                subject_id not in self.students[student_id]):
            return 0.0
        return self.students[student_id][subject_id].v_smoothed

    def get_aggregate_velocity(self, student_id: str) -> float:
        if student_id not in self.students:
            return 0.0
        velocities = []
        for state in self.students[student_id].values():
            if state.n_updates >= 2:
                velocities.append(state.v_smoothed)
        return float(np.mean(velocities)) if velocities else 0.0

    def get_consistency(self, student_id: str, window: int = 10) -> float:
        if student_id not in self.students:
            return 0.5
        max_var = 0.02  # mastery deltas swing ±0.07; generous for low-n subjects
        weighted_sum = 0.0
        total_n = 0
        for state in self.students[student_id].values():
            recent = state.velocity_history[-window:]
            n = len(recent)
            if n >= 2:
                c = max(0.0, 1.0 - np.var(recent) / max_var)
                weighted_sum += n * c
                total_n += n
        return float(weighted_sum / total_n) if total_n > 0 else 0.5

    def normalize_for_mvs(self, velocity: float) -> float:
        # Map [-0.02, +0.02] → [0, 1], symmetric: 0 velocity → 0.5
        return float(np.clip((velocity + 0.02) / 0.04, 0.0, 1.0))


# ── 3f. REGRESSION VELOCITY TRACKER (OLS slope) ─────────────────────

@dataclass
class RegressionSubjectState:
    """Per-student per-subject state for OLS regression velocity."""
    outcomes: List[int] = field(default_factory=list)  # binary correct/incorrect
    velocity_history: List[float] = field(default_factory=list)


class RegressionVelocityTracker(BaseVelocityTracker):
    """OLS regression slope of correctness over interaction number.

    Fits a simple linear regression y = a + b*x on the full outcome history
    per subject, where y ∈ {0, 1} and x = interaction index. The slope b
    captures the overall trend including gradual improvements that sliding-
    window methods (ZPDES) might miss.

    Range: slope is typically in [-0.05, +0.05] per interaction. Scaled by
    scale_factor=20 to map to [-1, +1] for ensemble compatibility.
    """

    def __init__(self, min_interactions: int = 4):
        self.min_interactions = min_interactions
        self.students: Dict[str, Dict[str, RegressionSubjectState]] = {}

    @property
    def name(self) -> str:
        return "Regression (OLS)"

    def _get_state(self, student_id: str, subject_id: str) -> RegressionSubjectState:
        if student_id not in self.students:
            self.students[student_id] = {}
        if subject_id not in self.students[student_id]:
            self.students[student_id][subject_id] = RegressionSubjectState()
        return self.students[student_id][subject_id]

    @staticmethod
    def _ols_slope(outcomes: List[int]) -> float:
        """Compute OLS slope efficiently: b = Cov(x,y) / Var(x)."""
        n = len(outcomes)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(outcomes, dtype=float)
        x_mean = (n - 1) / 2.0
        y_mean = y.mean()
        cov_xy = np.dot(x - x_mean, y - y_mean)
        var_x = np.dot(x - x_mean, x - x_mean)
        if var_x < 1e-12:
            return 0.0
        return float(cov_xy / var_x)

    def record(self, student_id: str, timestamp_days: float,
               overall_mastery: float,
               subject_id: Optional[str] = None,
               subject_mastery: Optional[float] = None,
               is_correct: Optional[bool] = None,
               kt_prediction: Optional[float] = None) -> Tuple[float, float]:
        if subject_id is None or is_correct is None:
            return 0.0, 0.0

        state = self._get_state(student_id, subject_id)
        state.outcomes.append(1 if is_correct else 0)

        n = len(state.outcomes)
        if n < self.min_interactions:
            state.velocity_history.append(0.0)
            return 0.0, 0.0

        slope = self._ols_slope(state.outcomes)
        state.velocity_history.append(slope)

        # Confidence: ramp from 0 → 1 over 16 interactions
        confidence = min(1.0, (n - self.min_interactions) / 12.0)
        return slope, confidence

    def get_subject_velocity(self, student_id: str, subject_id: str) -> float:
        if (student_id not in self.students or
                subject_id not in self.students[student_id]):
            return 0.0
        state = self.students[student_id][subject_id]
        if len(state.outcomes) < self.min_interactions:
            return 0.0
        return self._ols_slope(state.outcomes)

    def get_aggregate_velocity(self, student_id: str) -> float:
        if student_id not in self.students:
            return 0.0
        velocities = []
        for state in self.students[student_id].values():
            if len(state.outcomes) >= self.min_interactions:
                velocities.append(self._ols_slope(state.outcomes))
        return float(np.mean(velocities)) if velocities else 0.0

    def get_consistency(self, student_id: str, window: int = 10) -> float:
        if student_id not in self.students:
            return 0.5
        max_var = 0.05  # OLS slopes swing wide for low-n subjects
        weighted_sum = 0.0
        total_n = 0
        for state in self.students[student_id].values():
            recent = state.velocity_history[-window:]
            n = len(recent)
            if n >= 2:
                c = max(0.0, 1.0 - np.var(recent) / max_var)
                weighted_sum += n * c
                total_n += n
        return float(weighted_sum / total_n) if total_n > 0 else 0.5

    def normalize_for_mvs(self, velocity: float) -> float:
        # Map [-0.05, +0.05] → [0, 1], symmetric: 0 slope → 0.5
        return float(np.clip((velocity + 0.05) / 0.10, 0.0, 1.0))


# ── 3g. ENSEMBLE VELOCITY TRACKER ─────────────────────────────────────

@dataclass
class EnsembleSubjectState:
    """Lightweight bookkeeping for ensemble (actual state lives in child trackers)."""
    n_interactions: int = 0
    velocity_history: List[float] = field(default_factory=list)


class EnsembleVelocityTracker(BaseVelocityTracker):
    """Confidence-weighted ensemble of KT and Mastery Delta velocity signals.

    Both components are difficulty-aware:
      - KT: EMA-smoothed P(correct) deltas from SAINT-Lite (trained on IRT features).
            Works from the 2nd interaction — best cold-start signal.
      - Mastery Delta: EMA-smoothed EKF mastery deltas. Uses 2PL IRT difficulty,
            discrimination, and FSRS forgetting. Most principled signal once
            the EKF has converged.

    CUSUM and Regression were removed after ablation analysis: both operate on
    raw binary correctness (difficulty-unaware), adding noise for strong students
    and correlating highly with each other (r=0.53). Removing them improved
    velocity accuracy for stable students while maintaining sensitivity to
    genuine learning transitions.

    Weights adapted by data availability and modulated by each tracker's
    self-reported confidence.
    """

    # Scale factors to normalize each tracker's EMA-smoothed velocity to [-1, +1].
    # Calibrated on the actual EMA output range (not per-interaction deltas):
    #   KT: EMA of P_kt deltas, typical range [-0.05, +0.20] during transitions
    #   Mastery Delta: EMA of mastery deltas, typical range [-0.02, +0.08] during
    #     EKF convergence, near zero once converged
    SCALE_FACTORS = {
        'kt':            5.0,    # maps ~[-0.20, +0.20] → [-1, +1]
        'mastery_delta': 15.0,   # maps ~[-0.07, +0.07] → [-1, +1]
    }

    # Base weights: 2-component ensemble.
    # Both signals are difficulty-aware via EKF/IRT and SAINT-Lite.
    # Mastery Delta is the most principled signal (directly tracks EKF state
    # changes), while KT provides faster response and cold-start capability.
    BASE_WEIGHTS = {
        'kt':            0.400,  # KT model signal, difficulty-aware, fast cold start
        'mastery_delta': 0.600,  # EKF mastery changes — most principled signal
    }

    def __init__(self,
                 kt_tracker: Optional['KTVelocityTracker'] = None,
                 mastery_delta_tracker: Optional['MasteryDeltaVelocityTracker'] = None,
                 delegate_record: bool = True):
        """
        Args:
            delegate_record: If True, record() calls child trackers.
                If False (piped mode), children are recorded externally
                (e.g., by MultiVelocityPipeline).
        """
        self.trackers = {
            'kt': kt_tracker or KTVelocityTracker(ema_alpha=0.3),
            'mastery_delta': mastery_delta_tracker or MasteryDeltaVelocityTracker(ema_alpha=0.3),
        }
        self._delegate_record = delegate_record
        self.students: Dict[str, Dict[str, EnsembleSubjectState]] = {}

    @property
    def name(self) -> str:
        return "Ensemble (KT+MasteryDelta)"

    @staticmethod
    def _adaptive_multipliers(n: int) -> Dict[str, float]:
        """Weight multipliers based on data availability.

        Cold start (n<4): KT dominates (works from 2nd KT prediction).
        Medium (4-15): Balanced.
        Rich data (16+): Mastery Delta dominates (EKF has converged,
            difficulty-aware mastery changes are the most principled signal).
        """
        if n < 4:
            return {'kt': 2.0, 'mastery_delta': 1.0}
        elif n < 8:
            return {'kt': 1.5, 'mastery_delta': 1.2}
        elif n < 16:
            return {'kt': 1.0, 'mastery_delta': 1.0}
        else:
            return {'kt': 0.8, 'mastery_delta': 1.3}

    def _estimate_confidence(self, tracker_name: str, n: int) -> float:
        """Approximate tracker confidence from interaction count."""
        if tracker_name == 'kt':
            # KT: 0 → 0.5 → 1.0 after 0 → 1 → 2 updates
            return min(1.0, max(0.0, (n - 1) * 0.5)) if n >= 1 else 0.0
        elif tracker_name == 'mastery_delta':
            # Mastery delta: needs at least 2 interactions for a delta
            return min(1.0, max(0.0, (n - 1) / 8.0))
        else:
            return 0.0

    def _combine(self, student_id: str, subject_id: str,
                 velocities: Optional[Dict[str, float]] = None,
                 confidences: Optional[Dict[str, float]] = None,
                 ) -> Tuple[float, float]:
        """Core combination logic: weighted average on common scale."""
        state = self.students.get(student_id, {})
        subj = state.get(subject_id)
        n = subj.n_interactions if subj else 0
        multipliers = self._adaptive_multipliers(n)

        # Get velocities and confidences if not provided
        if velocities is None:
            velocities = {}
            for name, tracker in self.trackers.items():
                velocities[name] = tracker.get_subject_velocity(student_id, subject_id)
        if confidences is None:
            confidences = {}
            for name in self.trackers:
                confidences[name] = self._estimate_confidence(name, n)

        # Compute effective weights
        effective_w = {}
        for name in self.trackers:
            effective_w[name] = (self.BASE_WEIGHTS[name]
                                 * multipliers[name]
                                 * confidences[name])

        total_w = sum(effective_w.values())
        if total_w < 1e-10:
            return 0.0, 0.0

        # Scale velocities to [-1, +1] and combine
        ensemble_v = 0.0
        for name in self.trackers:
            v_scaled = np.clip(velocities[name] * self.SCALE_FACTORS[name], -1.0, 1.0)
            ensemble_v += effective_w[name] * v_scaled
        ensemble_v /= total_w

        # Ensemble confidence = weighted mean of component confidences
        base_total = sum(self.BASE_WEIGHTS[name] * multipliers[name]
                         for name in self.trackers)
        ensemble_conf = 0.0
        if base_total > 0:
            for name in self.trackers:
                w_contrib = self.BASE_WEIGHTS[name] * multipliers[name]
                ensemble_conf += (w_contrib / base_total) * confidences[name]

        return float(ensemble_v), float(ensemble_conf)

    def record(self, student_id: str, timestamp_days: float,
               overall_mastery: float,
               subject_id: Optional[str] = None,
               subject_mastery: Optional[float] = None,
               is_correct: Optional[bool] = None,
               kt_prediction: Optional[float] = None) -> Tuple[float, float]:
        if subject_id is None:
            return 0.0, 0.0

        # Delegate to children or read their state
        velocities = {}
        confidences = {}
        if self._delegate_record:
            for name, tracker in self.trackers.items():
                v, conf = tracker.record(
                    student_id=student_id, timestamp_days=timestamp_days,
                    overall_mastery=overall_mastery, subject_id=subject_id,
                    subject_mastery=subject_mastery, is_correct=is_correct,
                    kt_prediction=kt_prediction)
                velocities[name] = v
                confidences[name] = conf
        else:
            for name, tracker in self.trackers.items():
                velocities[name] = tracker.get_subject_velocity(student_id, subject_id)

        # Update bookkeeping
        if student_id not in self.students:
            self.students[student_id] = {}
        if subject_id not in self.students[student_id]:
            self.students[student_id][subject_id] = EnsembleSubjectState()
        subj_state = self.students[student_id][subject_id]
        subj_state.n_interactions += 1

        if not self._delegate_record:
            for name in self.trackers:
                confidences[name] = self._estimate_confidence(
                    name, subj_state.n_interactions)

        ens_v, ens_conf = self._combine(
            student_id, subject_id, velocities, confidences)
        subj_state.velocity_history.append(ens_v)
        return ens_v, ens_conf

    def get_subject_velocity(self, student_id: str, subject_id: str) -> float:
        v, _ = self._combine(student_id, subject_id)
        return v

    def get_aggregate_velocity(self, student_id: str) -> float:
        if student_id not in self.students:
            return 0.0
        velocities = []
        for subject_id, subj_state in self.students[student_id].items():
            if subj_state.n_interactions >= 2:
                velocities.append(self.get_subject_velocity(student_id, subject_id))
        return float(np.mean(velocities)) if velocities else 0.0

    def get_consistency(self, student_id: str, window: int = 10) -> float:
        """Weighted average of component consistencies."""
        if student_id not in self.students:
            return 0.5
        student_subjects = self.students[student_id]
        total_n = sum(s.n_interactions for s in student_subjects.values())
        avg_n = total_n // max(1, len(student_subjects))
        multipliers = self._adaptive_multipliers(avg_n)

        weights = {}
        for name in self.trackers:
            weights[name] = self.BASE_WEIGHTS[name] * multipliers[name]
        total_w = sum(weights.values())
        if total_w < 1e-10:
            return 0.5

        ensemble_c = 0.0
        for name, tracker in self.trackers.items():
            ensemble_c += weights[name] * tracker.get_consistency(student_id, window)
        return float(ensemble_c / total_w)

    def normalize_for_mvs(self, velocity: float) -> float:
        # Ensemble output is in [-1, +1] (common scale)
        return float(np.clip((velocity + 1.0) / 2.0, 0.0, 1.0))

    def get_component_velocities(self, student_id: str, subject_id: str
                                  ) -> Dict[str, Tuple[float, float, float]]:
        """Debug helper: {name: (raw_velocity, confidence, effective_weight)}."""
        if student_id not in self.students:
            return {}
        subj_state = self.students[student_id].get(subject_id)
        if subj_state is None:
            return {}
        n = subj_state.n_interactions
        multipliers = self._adaptive_multipliers(n)
        result = {}
        for name, tracker in self.trackers.items():
            v = tracker.get_subject_velocity(student_id, subject_id)
            conf = self._estimate_confidence(name, n)
            ew = self.BASE_WEIGHTS[name] * multipliers[name] * conf
            result[name] = (v, conf, ew)
        return result


# ═══════════════════════════════════════════════════════════════════════
# 4. MVS SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_mvs(mastery_level: float,
                velocity_smoothed: float,
                consistency: float,
                breadth: float,
                w_level: float = 40.0,
                w_velocity: float = 30.0,
                w_consistency: float = 15.0,
                w_breadth: float = 15.0) -> float:
    """Mastery Velocity Score (MVS).

    MVS = w_level·M + w_velocity·V̄ + w_consistency·C + w_breadth·B

    All inputs should be in [0, 1]. Output in [0, 100].
    """
    # Clamp inputs
    M = np.clip(mastery_level, 0, 1)
    V = np.clip(velocity_smoothed, 0, 1)  # velocity normalized to [0,1]
    C = np.clip(consistency, 0, 1)
    B = np.clip(breadth, 0, 1)

    return w_level * M + w_velocity * V + w_consistency * C + w_breadth * B


def compute_breadth(topic_masteries: Dict[str, float],
                    taxonomy: Dict[str, str],
                    min_mastery: float = 0.6) -> float:
    """Breadth = fraction of subjects with at least one topic above threshold.

    Returns:
        Breadth score [0, 1]
    """
    subjects = {}
    for topic_id, mastery in topic_masteries.items():
        subj = taxonomy.get(topic_id, 'Unknown')
        if subj not in subjects:
            subjects[subj] = False
        if mastery >= min_mastery:
            subjects[subj] = True

    if not subjects:
        return 0.0
    return sum(subjects.values()) / len(subjects)


def normalize_velocity(v_smoothed: float,
                       v_min: float = -0.02,
                       v_max: float = 0.02) -> float:
    """Normalize velocity to [0, 1] for MVS computation.

    Symmetric range: zero velocity maps to 0.5 (neutral).
    """
    return np.clip((v_smoothed - v_min) / max(v_max - v_min, 1e-9), 0, 1)


# ═══════════════════════════════════════════════════════════════════════
# 5. RECOMMENDATION SCORING — ITZS (Information-Theoretic ZPD Scoring)
#
# Literature basis:
#   - ZPD peak: Wilson et al. (2019) "The 85% Rule", Nature Communications
#   - Learning progress: Clément et al. (2015) "Multi-Armed Bandits for ITS", JEDM
#   - Information gain: Shannon entropy for exploration (CAT literature)
#   - Review scheduling: FSRS v5 retrievability sigmoid
#   - Novelty: decaying exploration bonus for unseen topics in ZPD
# ═══════════════════════════════════════════════════════════════════════


def zpd_score(p_correct: float) -> float:
    """Gaussian ZPD score peaking at optimal accuracy (Wilson et al. 2019).

    zpd(p) = exp(-0.5 × ((p - μ*) / σ)²)

    where μ* = 1 - erfc(1/√2)/2 ≈ 0.8413 is the analytically derived
    optimal accuracy for gradient-descent learners with Gaussian noise.

    Returns ~1.0 when P(correct) ≈ 0.84 (sweet spot), drops to ~0
    for very easy (p→1) or very hard (p→0) topics.
    """
    return float(np.exp(-0.5 * ((p_correct - ZPD_PEAK) / ZPD_SIGMA) ** 2))


def review_urgency_sigmoid(R: float) -> float:
    """Sigmoid review urgency around retrievability threshold.

    urgency(R) = 1 / (1 + exp(k × (R - R_threshold)))

    Returns ~1.0 when R << threshold (memory fading, needs review),
    ~0.0 when R >> threshold (memory strong, no urgency).
    """
    return float(1.0 / (1.0 + np.exp(REVIEW_SIGMOID_K * (R - REVIEW_R_THRESHOLD))))


def information_gain(p_correct: float) -> float:
    """Shannon binary entropy — measures uncertainty about student knowledge.

    H(p) = -p·log₂(p) - (1-p)·log₂(1-p)

    Maximized at p=0.5 (model most uncertain about student), zero at
    p=0 or p=1. Drives exploration toward topics where practicing
    provides the most diagnostic value.
    """
    p = np.clip(p_correct, 1e-10, 1.0 - 1e-10)
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


@dataclass
class TopicScore:
    """Recommendation score for one topic (ITZS: 5 orthogonal components)."""
    topic_id: str
    total_score: float = 0.0
    # Components
    review_urgency: float = 0.0
    expected_learning_gain: float = 0.0
    information_gain_val: float = 0.0
    exam_importance: float = 0.0
    novelty_bonus: float = 0.0
    # State
    mastery: float = 0.0
    zpd_score_val: float = 0.0
    kt_prediction: float = 0.0
    retrievability: float = 1.0
    n_interactions: int = 0

    # Legacy aliases for backward compat (validate_mastery.py, tests)
    @property
    def decay_urgency(self) -> float:
        return self.review_urgency

    @property
    def coverage_gap(self) -> float:
        return 1.0 - self.mastery


def score_topics_for_recommendation(
    student_id: str,
    mastery_tracker: MasteryTracker,
    current_ts: float,
    pyq_weights: Optional[Dict[str, float]] = None,
    kt_predictions: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[TopicScore]:
    """Score all topics using ITZS (Information-Theoretic ZPD Scoring).

    Five orthogonal components, each naturally in [0, 1]:

      1. Expected Learning Gain (ELG):
         zpd(P_kt) × (1 - mastery)
         Peaks when topic is at ~84% predicted accuracy AND has headroom.

      2. Review Urgency:
         sigmoid(R_threshold - R)
         High when FSRS retrievability is low (memory fading).
         Zero for unseen topics (no memory to decay).

      3. Information Gain:
         H(P_kt) × (1 - mastery)
         Shannon entropy of KT prediction, weighted by headroom.
         Drives exploration toward uncertain topics.

      4. Exam Importance:
         pyq_weight × (1 - mastery)
         Prioritizes exam-relevant un-mastered topics.

      5. Novelty Bonus:
         zpd(P_kt) × max(0, 1 - n/N_decay)
         Encourages trying unseen topics, but only if in the ZPD.
         Decays to 0 after NOVELTY_DECAY_N interactions.

    Args:
        student_id: student identifier
        mastery_tracker: MasteryTracker with current state
        current_ts: current timestamp in days
        pyq_weights: {topic_id: exam_frequency_weight}
        kt_predictions: {topic_id: P(correct)} from KT model (e.g., SAINT-Lite).
            If None, uses mastery as P(correct) estimate.
        weights: override ITZS_WEIGHTS defaults

    Returns:
        List[TopicScore] sorted by total_score descending
    """
    w = {**ITZS_WEIGHTS, **(weights or {})}
    pwts = pyq_weights or mastery_tracker.pyq_weights
    kt = kt_predictions or {}
    scores = []

    # Normalize PYQ weights to [0, 1]
    max_pyq = max(pwts.values()) if pwts else 1.0
    max_pyq = max(max_pyq, 1e-9)

    for topic_id in mastery_tracker.taxonomy:
        ts = TopicScore(topic_id=topic_id)

        # ── Current mastery (with FSRS decay) ──
        m = mastery_tracker.get_topic_mastery(student_id, topic_id, current_ts)
        ts.mastery = m

        # ── Topic state for FSRS + interaction count ──
        # Read-only lookup — does NOT create phantom states for unseen topics
        topic_state = mastery_tracker._peek_topic_state(student_id, topic_id)
        n_interactions = topic_state.n_interactions if topic_state else 0
        ts.n_interactions = n_interactions

        # ── P(correct): KT prediction if available, else mastery estimate ──
        p = kt.get(topic_id, m)
        ts.kt_prediction = p

        # ── Component 1: Expected Learning Gain ──
        # zpd(p) peaks at P(correct) = 0.8413 (85% rule, Wilson et al.)
        # ELG = zpd(p) × (1 - mastery): high ZPD + room for growth
        ts.zpd_score_val = zpd_score(p)
        ts.expected_learning_gain = ts.zpd_score_val * (1.0 - m)

        # ── Component 2: Review Urgency ──
        # Sigmoid around R=0.70. Only for studied topics.
        # Unseen topics have no memory to decay → urgency = 0.
        if (topic_state is not None
                and topic_state.fsrs_state.stability > 0
                and n_interactions > 0):
            t_days = max(0, current_ts - topic_state.fsrs_state.last_review_ts)
            R = mastery_tracker.fsrs.retrievability(t_days, topic_state.fsrs_state.stability)
            ts.retrievability = R
            ts.review_urgency = review_urgency_sigmoid(R)
        else:
            ts.retrievability = 1.0
            ts.review_urgency = 0.0

        # ── Component 3: Information Gain ──
        # Shannon entropy of KT prediction, weighted by mastery headroom.
        # High when model is uncertain (p ≈ 0.5) AND topic isn't mastered.
        ts.information_gain_val = information_gain(p) * (1.0 - m)

        # ── Component 4: Exam Importance ──
        # PYQ weight × mastery headroom. Falls back to uniform when no PYQ data.
        pyq_w = pwts.get(topic_id, 0.0) / max_pyq
        ts.exam_importance = pyq_w * (1.0 - m)

        # ── Component 5: Novelty Bonus ──
        # zpd(p) × decay. Encourages exploration of unseen ZPD-ready topics.
        if n_interactions < NOVELTY_DECAY_N:
            ts.novelty_bonus = ts.zpd_score_val * (
                1.0 - n_interactions / NOVELTY_DECAY_N)
        else:
            ts.novelty_bonus = 0.0

        # ── Composite ITZS Score ──
        ts.total_score = (
            w['w_elg']     * ts.expected_learning_gain
            + w['w_review']  * ts.review_urgency
            + w['w_info']    * ts.information_gain_val
            + w['w_exam']    * ts.exam_importance
            + w['w_novelty'] * ts.novelty_bonus
        )

        scores.append(ts)

    scores.sort(key=lambda s: s.total_score, reverse=True)
    return scores


# ═══════════════════════════════════════════════════════════════════════
# 6. FULL PIPELINE — convenience function
# ═══════════════════════════════════════════════════════════════════════

class MasteryVelocityPipeline:
    """Full pipeline: process interactions → mastery → velocity → MVS → recommendations."""

    def __init__(self, taxonomy: Dict[str, str],
                 pyq_weights: Optional[Dict[str, float]] = None,
                 fsrs_params: Optional[Dict[str, float]] = None,
                 tracker: str = 'kalman'):
        """
        Args:
            tracker: 'kalman' (EKF, default) or 'beta' (Beta-Bayesian).
        """
        if tracker == 'kalman':
            self.mastery = KalmanMasteryTracker(taxonomy, pyq_weights, fsrs_params)
        else:
            self.mastery = MasteryTracker(taxonomy, pyq_weights, fsrs_params)
        self.velocity = EnsembleVelocityTracker()
        self.taxonomy = taxonomy

    def process_interaction(self, student_id: str, topic_id: str,
                            is_correct: bool, timestamp_days: float,
                            kt_prediction: Optional[float] = None,
                            discrimination: float = 1.0,
                            difficulty: float = 0.0) -> dict:
        """Process one interaction through the full pipeline.

        Args:
            kt_prediction: optional KT model P(correct). For Kalman tracker,
                used to initialize theta on first interaction (cold start).
                For Beta tracker, sets the Beta prior mean.
            discrimination: 2PL IRT discrimination (a parameter).
            difficulty: 2PL IRT difficulty (b parameter, logit scale).

        Returns dict with all computed values.
        """
        # 1. Update mastery (EKF or Beta-Bayesian)
        topic_m = self.mastery.update(student_id, topic_id, is_correct,
                                      timestamp_days, kt_prediction,
                                      discrimination=discrimination,
                                      difficulty=difficulty)

        # 2. Compute overall mastery and subject mastery
        overall_m = self.mastery.get_overall_mastery(student_id, timestamp_days)
        subject_id = self.taxonomy.get(topic_id)
        subject_m = self.mastery.get_subject_mastery(student_id, subject_id, timestamp_days)

        # 3. Update velocity with per-subject tracking (ensemble)
        v_ens, v_conf = self.velocity.record(
            student_id=student_id, timestamp_days=timestamp_days,
            overall_mastery=overall_m, subject_id=subject_id,
            subject_mastery=subject_m, is_correct=is_correct,
            kt_prediction=kt_prediction)

        # 4. Get topic state for FSRS info
        ts = self.mastery._get_topic_state(student_id, topic_id)

        return {
            'topic_mastery': topic_m,
            'overall_mastery': overall_m,
            'velocity': v_ens,
            'velocity_confidence': v_conf,
            'fsrs_stability': ts.fsrs_state.stability,
            'fsrs_difficulty': ts.fsrs_state.difficulty,
            'fsrs_retrievability': self.mastery.fsrs.retrievability(
                0, ts.fsrs_state.stability),  # R at review time = 1.0
        }

    def get_mvs(self, student_id: str,
                current_ts: Optional[float] = None) -> dict:
        """Compute full MVS score for a student."""
        overall_m = self.mastery.get_overall_mastery(student_id, current_ts)
        v_raw = self.velocity.get_aggregate_velocity(student_id)
        consistency = self.velocity.get_consistency(student_id)
        topic_ms = self.mastery.get_all_topic_masteries(student_id, current_ts)
        breadth = compute_breadth(topic_ms, self.taxonomy)
        v_norm = self.velocity.normalize_for_mvs(v_raw)
        mvs = compute_mvs(overall_m, v_norm, consistency, breadth)

        return {
            'mvs': mvs,
            'mastery_level': overall_m,
            'velocity_normalized': v_norm,
            'velocity_raw': v_raw,
            'consistency': consistency,
            'breadth': breadth,
            'topic_masteries': topic_ms,
        }

    def get_recommendations(self, student_id: str,
                            current_ts: float, top_n: int = 10,
                            kt_predictions: Optional[Dict[str, float]] = None,
                            ) -> List[TopicScore]:
        """Get ranked topic recommendations using ITZS scoring.

        Args:
            kt_predictions: {topic_id: P(correct)} from KT model.
                If provided, enables ZPD targeting and information gain.
                If None, falls back to mastery-based estimates.
        """
        return score_topics_for_recommendation(
            student_id, self.mastery, current_ts,
            kt_predictions=kt_predictions,
        )[:top_n]


# ═══════════════════════════════════════════════════════════════════════
# 7. MULTI-VELOCITY COMPARISON PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class MultiVelocityPipeline:
    """Wrapper that runs multiple velocity trackers in parallel for comparison.

    Delegates mastery tracking to a single MasteryVelocityPipeline and
    feeds the same interaction data to all velocity trackers.
    """

    def __init__(self, taxonomy: Dict[str, str],
                 pyq_weights: Optional[Dict[str, float]] = None,
                 fsrs_params: Optional[Dict[str, float]] = None,
                 tracker: str = 'kalman'):
        self.base_pipeline = MasteryVelocityPipeline(
            taxonomy, pyq_weights, fsrs_params, tracker=tracker)
        self.taxonomy = taxonomy

        # Create shared tracker instances
        zpdes = ZPDESVelocityTracker(window_d=16)
        kt = KTVelocityTracker(ema_alpha=0.3)
        mastery_delta = MasteryDeltaVelocityTracker(ema_alpha=0.3)

        # All trackers kept for display/comparison, including ZPDES
        self.trackers: Dict[str, BaseVelocityTracker] = {
            'zpdes': zpdes, 'kt': kt,
            'mastery_delta': mastery_delta,
        }
        # Ensemble uses KT + Mastery Delta (both difficulty-aware)
        self.ensemble = EnsembleVelocityTracker(
            kt, mastery_delta, delegate_record=False)

    def process_interaction(self, student_id: str, topic_id: str,
                            is_correct: bool, timestamp_days: float,
                            kt_prediction: Optional[float] = None,
                            discrimination: float = 1.0,
                            difficulty: float = 0.0) -> dict:
        """Process interaction through base pipeline + all velocity trackers."""
        # 1. Run the base pipeline (mastery + baseline velocity)
        result = self.base_pipeline.process_interaction(
            student_id, topic_id, is_correct, timestamp_days, kt_prediction,
            discrimination=discrimination, difficulty=difficulty)

        # 2. Feed the same data to the three new trackers
        # With identity taxonomy (subject==topic), topic_mastery IS subject mastery.
        # Only the baseline VelocityTracker uses this value; ZPDES/CUSUM use
        # is_correct and KT uses kt_prediction.
        subject_id = self.taxonomy.get(topic_id)
        overall_m = result['overall_mastery']
        subject_m = result['topic_mastery']

        velocities = {'baseline': result['velocity']}

        for name, tracker in self.trackers.items():
            v, conf = tracker.record(
                student_id=student_id,
                timestamp_days=timestamp_days,
                overall_mastery=overall_m,
                subject_id=subject_id,
                subject_mastery=subject_m,
                is_correct=is_correct,
                kt_prediction=kt_prediction,
            )
            velocities[name] = v

        # 3. Ensemble reads children's state (piped mode)
        ens_v, ens_conf = self.ensemble.record(
            student_id=student_id,
            timestamp_days=timestamp_days,
            overall_mastery=overall_m,
            subject_id=subject_id,
            subject_mastery=subject_m,
            is_correct=is_correct,
            kt_prediction=kt_prediction,
        )
        velocities['ensemble'] = ens_v

        result['velocities'] = velocities
        return result

    def get_mvs_all(self, student_id: str,
                    current_ts: Optional[float] = None) -> Dict[str, dict]:
        """Compute MVS using each velocity approach.

        Returns dict of {approach_name: mvs_dict}.
        """
        overall_m = self.base_pipeline.mastery.get_overall_mastery(
            student_id, current_ts)
        topic_ms = self.base_pipeline.mastery.get_all_topic_masteries(
            student_id, current_ts)
        breadth = compute_breadth(topic_ms, self.taxonomy)

        results = {}

        # Baseline (base pipeline's own ensemble tracker)
        v_baseline = self.base_pipeline.velocity.get_aggregate_velocity(student_id)
        baseline_consistency = self.base_pipeline.velocity.get_consistency(
            student_id)
        v_norm_baseline = self.base_pipeline.velocity.normalize_for_mvs(v_baseline)
        mvs_baseline = compute_mvs(overall_m, v_norm_baseline,
                                   baseline_consistency, breadth)
        results['baseline'] = {
            'mvs': mvs_baseline,
            'velocity_raw': v_baseline,
            'velocity_normalized': v_norm_baseline,
            'consistency': baseline_consistency,
        }

        # New trackers
        for name, tracker in self.trackers.items():
            v_agg = tracker.get_aggregate_velocity(student_id)
            v_norm = tracker.normalize_for_mvs(v_agg)
            consistency = tracker.get_consistency(student_id)
            mvs = compute_mvs(overall_m, v_norm, consistency, breadth)
            results[name] = {
                'mvs': mvs,
                'velocity_raw': v_agg,
                'velocity_normalized': v_norm,
                'consistency': consistency,
            }

        # Ensemble
        ens_v_agg = self.ensemble.get_aggregate_velocity(student_id)
        ens_v_norm = self.ensemble.normalize_for_mvs(ens_v_agg)
        ens_consistency = self.ensemble.get_consistency(student_id)
        ens_mvs = compute_mvs(overall_m, ens_v_norm, ens_consistency, breadth)
        results['ensemble'] = {
            'mvs': ens_mvs,
            'velocity_raw': ens_v_agg,
            'velocity_normalized': ens_v_norm,
            'consistency': ens_consistency,
        }

        return results
