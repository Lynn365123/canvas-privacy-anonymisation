Add privacy anonymisation experiment
#!/usr/bin/env python3
"""Privacy experiment for a synthetic Canvas-style student dataset.

The program:
1. Generates synthetic data, or reads an existing CSV.
2. Measures baseline re-identification risk using equivalence classes.
3. Generalises quasi-identifiers.
4. Suppresses groups that fail k-anonymity.
5. Suppresses groups that fail distinct l-diversity.
6. Exports anonymised data, metrics and publication-ready figures.

This is an educational experiment, not proof that a real dataset is safe to
release. Release decisions also require a contextual privacy-risk assessment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIRECT_IDENTIFIER = "Student_ID"
ORIGINAL_QIS = ["Age", "Course", "Language", "Activity_Period"]
GENERALISED_QIS = [
    "Age_Band",
    "Course_Group",
    "Language_Group",
    "Activity_Period",
]
SENSITIVE_ATTRIBUTE = "Support_Category"


@dataclass(frozen=True)
class PrivacyMetrics:
    original_records: int
    baseline_singleton_records: int
    baseline_singleton_rate: float
    k: int
    l: int
    records_after_k: int
    records_after_k_and_l: int
    suppressed_for_k: int
    suppressed_for_l: int
    overall_suppressed: int
    overall_suppression_rate: float
    minimum_equivalence_class_size: int
    minimum_sensitive_value_diversity: int
    utility_total_variation_distance: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "Original records": self.original_records,
            "Baseline singleton records": self.baseline_singleton_records,
            "Baseline singleton rate (%)": round(self.baseline_singleton_rate * 100, 2),
            "k value": self.k,
            "l value": self.l,
            "Records after k-anonymity": self.records_after_k,
            "Records after k-anonymity and l-diversity": self.records_after_k_and_l,
            "Records suppressed for k-anonymity": self.suppressed_for_k,
            "Additional records suppressed for l-diversity": self.suppressed_for_l,
            "Overall records suppressed": self.overall_suppressed,
            "Overall suppression rate (%)": round(self.overall_suppression_rate * 100, 2),
            "Minimum equivalence-class size": self.minimum_equivalence_class_size,
            "Minimum sensitive-value diversity": self.minimum_sensitive_value_diversity,
            "Utility total variation distance": round(
                self.utility_total_variation_distance, 4
            ),
        }


def generate_synthetic_data(number_of_records: int, seed: int) -> pd.DataFrame:
    """Generate fictional student data without using real personal information."""
    rng = np.random.default_rng(seed)

    courses = [
        "Cybersecurity",
        "Information Technology",
        "Engineering",
        "Business",
        "Nursing",
        "Design",
    ]
    languages = ["English", "Mandarin", "Arabic", "Vietnamese"]
    activity_periods = ["Morning", "Afternoon", "Evening", "Night"]
    support_categories = [
        "None",
        "Academic",
        "Financial",
        "Accessibility",
        "Wellbeing",
