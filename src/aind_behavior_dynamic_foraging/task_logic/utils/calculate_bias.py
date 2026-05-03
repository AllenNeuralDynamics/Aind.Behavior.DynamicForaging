import logging
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

logger = logging.getLogger(__name__)


def calculate_bias(outcomes: List[TrialOutcome]) -> float:
    """Estimate the side bias of an animal using logistic regression on recent trial history.

    Fits a Su2022-style logistic regression model using rewarded and unrewarded choice
    history as predictors. The intercept of the fitted model is returned as the bias,
    representing the animal's baseline tendency to choose right independent of reward history.

    Parameters
    ----------
    outcomes : List[TrialOutcome]
        List of trial outcomes. Auto-response and ignored trials are excluded.
        Only the most recent 200 trials are used.

    Returns
    -------
    float
        The logistic regression intercept, representing side bias.
        Positive values indicate a bias toward right, negative toward left.
    """

    solver = "liblinear"
    l1_ratios = (0,)
    lag = 5
    cv = 10
    cs = 10

    # reduce outcomes to last 200
    outcomes = outcomes[-200:]

    # exclude auto response and ignored trials
    filtered = [t for t in outcomes if t.is_right_choice is not None and t.trial.is_auto_response_right is None]

    if len(filtered) <= lag:
        logger.warning("Not enough choices to calculate bias.")
        return np.nan

    is_right_choice_history = np.array([t.is_right_choice for t in filtered], dtype=float)
    is_rewarded_history = np.array([t.is_rewarded for t in filtered], dtype=float)

    # transform to +-1 space for zero centered logistic regression
    choice_signed = 2 * is_right_choice_history - 1  # left=0 → -1, right=1 → +1
    reward_signed = 2 * is_rewarded_history - 1  # unrewarded=0 → -1, rewarded=1 → +1

    rewarded_choice = choice_signed * (reward_signed == 1)
    unrewarded_choice = choice_signed * (reward_signed == -1)

    trial_length = len(is_rewarded_history)
    x = np.zeros((trial_length - lag, 2 * lag))
    for i in range(lag, trial_length):
        x[i - lag] = np.hstack([choice[i - lag : i] for choice in [rewarded_choice, unrewarded_choice]])

    y = choice_signed[lag:]

    if len(np.unique(y)) < 2:  # all choices are the same, return max bias in that direction
        logger.warning("All choices are the same, cannot calculate bias.")
        return np.nan

    logistic_reg = LogisticRegressionCV(
        solver=solver,
        l1_ratios=l1_ratios,
        Cs=cs,
        cv=cv,
        use_legacy_attributes=True,
    )
    logistic_reg.fit(x, y)

    bias = logistic_reg.intercept_[0]
    return bias
