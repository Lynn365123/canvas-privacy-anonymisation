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
    ]

    # Probabilities are fictional and are used only to make the experiment
    # reproducible. They are not estimates of the UTS student population.
    return pd.DataFrame(
        {
            DIRECT_IDENTIFIER: [f"S{i:05d}" for i in range(1, number_of_records + 1)],
            "Age": rng.integers(18, 39, size=number_of_records),
            "Course": rng.choice(
                courses,
                size=number_of_records,
                p=[0.18, 0.20, 0.17, 0.18, 0.15, 0.12],
            ),
            "Language": rng.choice(
                languages,
                size=number_of_records,
                p=[0.50, 0.23, 0.15, 0.12],
            ),
            "Activity_Period": rng.choice(
                activity_periods,
                size=number_of_records,
                p=[0.24, 0.28, 0.38, 0.10],
            ),
            SENSITIVE_ATTRIBUTE: rng.choice(
                support_categories,
                size=number_of_records,
                p=[0.48, 0.20, 0.10, 0.09, 0.13],
            ),
        }
    )


def validate_columns(data: pd.DataFrame) -> None:
    required = set([DIRECT_IDENTIFIER, *ORIGINAL_QIS, SENSITIVE_ATTRIBUTE])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Input data are missing required columns: {missing}")
    if data.empty:
        raise ValueError("Input data contain no records.")
    if data[list(required)].isna().any().any():
        raise ValueError("Required columns contain missing values.")


def age_band(age: int) -> str:
    lower = (int(age) // 10) * 10
    return f"{lower}-{lower + 9}"


def course_group(course: str) -> str:
    mapping = {
        "Cybersecurity": "Technology",
        "Information Technology": "Technology",
        "Engineering": "Engineering and Design",
        "Design": "Engineering and Design",
        "Business": "Business",
        "Nursing": "Health",
    }
    return mapping.get(str(course), "Other discipline")


def language_group(language: str) -> str:
    return "English" if str(language).strip().lower() == "english" else "Other language"


def generalise(data: pd.DataFrame) -> pd.DataFrame:
    transformed = data.copy()
    transformed["Age_Band"] = transformed["Age"].map(age_band)
    transformed["Course_Group"] = transformed["Course"].map(course_group)
    transformed["Language_Group"] = transformed["Language"].map(language_group)
    return transformed


def class_sizes(data: pd.DataFrame, qis: Iterable[str]) -> pd.Series:
    """Return one row per equivalence class and its number of records."""
    return data.groupby(list(qis), dropna=False).size().rename("Class_Size")


def attach_class_size(data: pd.DataFrame, qis: Iterable[str]) -> pd.Series:
    """Return the equivalence-class size aligned to every record."""
    return data.groupby(list(qis), dropna=False)[list(qis)[0]].transform("size")


def apply_k_anonymity(
    data: pd.DataFrame, qis: list[str], k_value: int
) -> pd.DataFrame:
    sizes = attach_class_size(data, qis)
    return data.loc[sizes >= k_value].copy()


def apply_l_diversity(
    data: pd.DataFrame,
    qis: list[str],
    sensitive_attribute: str,
    l_value: int,
) -> pd.DataFrame:
    diversity = data.groupby(qis, dropna=False)[sensitive_attribute].nunique()
    valid_groups = diversity.loc[diversity >= l_value].index

    indexed = data.set_index(qis)
    result = indexed.loc[indexed.index.isin(valid_groups)].reset_index()
    return result


def minimum_class_size(data: pd.DataFrame, qis: list[str]) -> int:
    if data.empty:
        return 0
    return int(class_sizes(data, qis).min())


def minimum_diversity(
    data: pd.DataFrame, qis: list[str], sensitive_attribute: str
) -> int:
    if data.empty:
        return 0
    values = data.groupby(qis, dropna=False)[sensitive_attribute].nunique()
    return int(values.min())


def total_variation_distance(
    before: pd.DataFrame, after: pd.DataFrame
) -> float:
    """Measure distortion in a representative Course x Activity query."""
    dimensions = ["Course_Group", "Activity_Period"]
    p = before.groupby(dimensions).size().div(len(before))
    q = after.groupby(dimensions).size().div(len(after)) if len(after) else pd.Series(dtype=float)
    combined = pd.concat([p.rename("before"), q.rename("after")], axis=1).fillna(0)
    return float(0.5 * (combined["before"] - combined["after"]).abs().sum())


def calculate_metrics(
    original: pd.DataFrame,
    generalised: pd.DataFrame,
    after_k: pd.DataFrame,
    after_l: pd.DataFrame,
    k_value: int,
    l_value: int,
) -> PrivacyMetrics:
    baseline_sizes = attach_class_size(original, ORIGINAL_QIS)
    singleton_records = int((baseline_sizes == 1).sum())
    original_count = len(original)
    final_count = len(after_l)

    return PrivacyMetrics(
        original_records=original_count,
        baseline_singleton_records=singleton_records,
        baseline_singleton_rate=singleton_records / original_count,
        k=k_value,
        l=l_value,
        records_after_k=len(after_k),
        records_after_k_and_l=final_count,
        suppressed_for_k=original_count - len(after_k),
        suppressed_for_l=len(after_k) - final_count,
        overall_suppressed=original_count - final_count,
        overall_suppression_rate=(original_count - final_count) / original_count,
        minimum_equivalence_class_size=minimum_class_size(after_l, GENERALISED_QIS),
        minimum_sensitive_value_diversity=minimum_diversity(
            after_l, GENERALISED_QIS, SENSITIVE_ATTRIBUTE
        ),
        utility_total_variation_distance=total_variation_distance(
            generalised, after_l
        ),
    )


def save_comparison_figure(
    original: pd.DataFrame,
    final_data: pd.DataFrame,
    output_path: Path,
    k_value: int,
    l_value: int,
) -> None:
    before = class_sizes(original, ORIGINAL_QIS)
    after = class_sizes(final_data, GENERALISED_QIS)

    figure, axes = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True)
    axes[0].hist(
        before,
        bins=np.arange(0.5, max(int(before.max()), 1) + 1.5, 1),
        color="#2878B5",
        edgecolor="white",
    )
    axes[0].set_title("(a) Before anonymisation", loc="left", fontweight="bold")
    axes[0].set_xlabel("Equivalence-class size")
    axes[0].set_ylabel("Number of equivalence classes")

    if not after.empty:
        bins = np.arange(max(4.5, after.min() - 0.5), after.max() + 1.5, 1)
        axes[1].hist(after, bins=bins, color="#2878B5", edgecolor="white")
        axes[1].axvline(
            k_value,
            color="#C23B22",
            linestyle="--",
            linewidth=1.5,
            label=f"k = {k_value}",
        )
        axes[1].legend(frameon=False)
    axes[1].set_title(
        f"(b) After applying k = {k_value} and l = {l_value}",
        loc="left",
        fontweight="bold",
    )
    axes[1].set_xlabel("Equivalence-class size")
    axes[1].set_ylabel("Number of equivalence classes")

    figure.suptitle(
        "Equivalence-Class Size Distributions Before and After Privacy Controls",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_l_diversity_figure(final_data: pd.DataFrame, output_path: Path) -> None:
    diversity = (
        final_data.groupby(GENERALISED_QIS, dropna=False)[SENSITIVE_ATTRIBUTE]
        .nunique()
        .sort_values()
    )
    counts = diversity.value_counts().sort_index()

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.bar(counts.index.astype(str), counts.values, color="#3B8C6E")
    axis.set_title("Sensitive-Value Diversity in Retained Equivalence Classes")
    axis.set_xlabel("Distinct Support Category values per class")
    axis.set_ylabel("Number of equivalence classes")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_experiment(
    data: pd.DataFrame, output_directory: Path, k_value: int, l_value: int
) -> PrivacyMetrics:
    validate_columns(data)
    output_directory.mkdir(parents=True, exist_ok=True)

    generalised = generalise(data)
    after_k = apply_k_anonymity(generalised, GENERALISED_QIS, k_value)
    after_l = apply_l_diversity(
        after_k, GENERALISED_QIS, SENSITIVE_ATTRIBUTE, l_value
    )

    if after_l.empty:
        raise RuntimeError(
            "No records satisfied the selected k and l thresholds. "
            "Review the thresholds or generalisation strategy."
        )

    metrics = calculate_metrics(
        data, generalised, after_k, after_l, k_value, l_value
    )

    # Release only generalised quasi-identifiers and the sensitive attribute.
    release_columns = [*GENERALISED_QIS, SENSITIVE_ATTRIBUTE]
    after_l[release_columns].to_csv(
        output_directory / f"anonymised_k{k_value}_l{l_value}.csv", index=False
    )
    pd.DataFrame(
        list(metrics.as_dict().items()), columns=["Metric", "Result"]
    ).to_csv(output_directory / "privacy_metrics.csv", index=False)

    save_comparison_figure(
        data,
        after_l,
        output_directory / "equivalence_class_comparison.png",
        k_value,
        l_value,
    )
    save_l_diversity_figure(
        after_l, output_directory / "l_diversity_distribution.png"
    )

    return metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate k-anonymity and l-diversity in Canvas-style student data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional input CSV. If omitted, synthetic data are generated.",
    )
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("privacy_experiment_output")
    )
    arguments = parser.parse_args()
    if arguments.records < 1 or arguments.k < 2 or arguments.l < 2:
        parser.error("records must be positive; k and l must each be at least 2")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    if arguments.input:
        data = pd.read_csv(arguments.input)
    else:
        data = generate_synthetic_data(arguments.records, arguments.seed)
        arguments.output.mkdir(parents=True, exist_ok=True)
        data.to_csv(arguments.output / "synthetic_canvas_data.csv", index=False)

    metrics = run_experiment(data, arguments.output, arguments.k, arguments.l)

    print("\nPrivacy experiment completed\n" + "-" * 32)
    for name, value in metrics.as_dict().items():
        print(f"{name}: {value}")
    print(f"\nOutputs saved to: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
