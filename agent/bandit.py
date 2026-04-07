"""
agent/bandit.py
===============
Thompson Sampling multi-armed bandit for signal selection.

Each signal is modeled as a Beta distribution:
  - alpha: number of "good" outcomes (signal helped resolve uncertainty)
  - beta: number of "bad" outcomes (signal didn't help)

Prior: Beta(1, 1) = Uniform distribution
Updates:
  - reward = 1 (positive) → alpha += 1
  - reward = 0 (negative) → beta += 1

Selection: sample from Beta(alpha, beta) for each signal,
multiply by VOI to get adjusted priority.
"""

from __future__ import annotations

import json
import os
import numpy as np
from typing import Optional


class ThompsonBandit:
    """
    Thompson Sampling bandit for adaptive signal selection.

    Each arm (signal) maintains a Beta distribution parameterized by
    (alpha, beta) counts of positive/negative outcomes.

    The agent samples from each distribution and uses the sample as
    a multiplicative bonus for VOI-based selection.
    """

    PRIOR_ALPHA = 1
    PRIOR_BETA = 1

    def __init__(self):
        self._arms: dict[str, dict[str, int]] = {}

    def init_arm(self, signal_name: str) -> None:
        """Initialize a new arm with Beta(1, 1) prior."""
        if signal_name not in self._arms:
            self._arms[signal_name] = {
                "alpha": self.PRIOR_ALPHA,
                "beta": self.PRIOR_BETA,
            }

    def sample(self, signal_name: str) -> float:
        """
        Sample from the posterior Beta distribution for this signal.

        Parameters
        ----------
        signal_name : str
            Name of the signal arm.

        Returns
        -------
        float
            A sample from Beta(alpha, beta).
        """
        self.init_arm(signal_name)
        arm = self._arms[signal_name]
        return np.random.beta(arm["alpha"], arm["beta"])

    def get_priority(self, signal_name: str) -> float:
        """
        Returns the current priority score for a signal.
        This is just a wrapper around sample().

        Parameters
        ----------
        signal_name : str
            Name of the signal arm.

        Returns
        -------
        float
            Priority score (sample from Beta distribution).
        """
        return self.sample(signal_name)

    def update(self, signal_name: str, reward: int) -> None:
        """
        Update the arm's Beta distribution based on reward.

        Parameters
        ----------
        signal_name : str
            Name of the signal arm.
        reward : int
            1 = positive outcome (signal helped resolve uncertainty)
            0 = negative outcome (signal didn't help)
        """
        self.init_arm(signal_name)
        arm = self._arms[signal_name]

        if reward == 1:
            arm["alpha"] += 1
        elif reward == 0:
            arm["beta"] += 1

    def get_stats(self, signal_name: str) -> dict:
        """
        Returns the current statistics for a signal arm.

        Parameters
        ----------
        signal_name : str
            Name of the signal arm.

        Returns
        -------
        dict
            Contains alpha, beta, and estimated success rate.
        """
        self.init_arm(signal_name)
        arm = self._arms[signal_name]
        total = arm["alpha"] + arm["beta"]
        return {
            "alpha": arm["alpha"],
            "beta": arm["beta"],
            "success_rate": arm["alpha"] / total if total > 0 else 0.5,
            "n_trials": total,
        }

    def save_state(self, path: str) -> None:
        """
        Persist the bandit state to a JSON file.

        Parameters
        ----------
        path : str
            File path to save the state.
        """
        state = {
            "arms": {
                name: {
                    "alpha": arm["alpha"],
                    "beta": arm["beta"],
                }
                for name, arm in self._arms.items()
            }
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        """
        Load the bandit state from a JSON file.

        Parameters
        ----------
        path : str
            File path to load the state from.
        """
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            state = json.load(f)

        self._arms = {
            name: {
                "alpha": data["alpha"],
                "beta": data["beta"],
            }
            for name, data in state.get("arms", {}).items()
        }

    def reset(self) -> None:
        """Reset all arms to the prior Beta(1, 1) distribution."""
        self._arms.clear()
