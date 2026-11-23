
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"


def save_plot(fig, filename, save=True):
    """Save a visualization when requested."""

    if save:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            PLOT_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )


def plot_attrition_distribution(df, save=True):
    """Visualize the distribution of employees who left and stayed."""

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Attrition",
        hue="Attrition",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Employee Attrition Distribution")
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Number of Employees")

    fig.tight_layout()
    save_plot(fig, "attrition_distribution_modular.png", save)

    return fig


def plot_correlation_heatmap(df, categorical_cols, save=True):
    """Visualize correlations between encoded numerical features."""

    data = df.copy()

    data["Attrition"] = data["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    # Remove identifiers, metadata and constant features
    data.drop(
        [
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours",
            "Over18"
        ],
        axis=1,
        inplace=True,
        errors="ignore"
    )

    constant_columns = [
        column
        for column in data.columns
        if column != "Attrition"
        and data[column].nunique(dropna=False) <= 1
    ]

    data.drop(columns=constant_columns, inplace=True)

    encoded_df = pd.get_dummies(
        data,
        columns=[
            col for col in categorical_cols
            if col in data.columns
        ],
        drop_first=True,
        dtype=int
    )

    correlation = encoded_df.corr()

    fig, ax = plt.subplots(figsize=(16, 10))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        annot=False,
        center=0,
        ax=ax
    )

    ax.set_title("Feature Correlation Heatmap")

    fig.tight_layout()
    save_plot(fig, "correlation_heatmap_modular.png", save)

    return fig


def plot_attrition_by_department(df, save=True):
    """Compare attrition rates across departments."""

    data = df.copy()
    data["Attrition_Flag"] = (
        data["Attrition"] == "Yes"
    ).astype(int)

    rates = (
        data.groupby("Department")["Attrition_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="Attrition Rate")
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=rates,
        x="Department",
        y="Attrition Rate",
        hue="Department",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Attrition Rate by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Employees Who Left (%)")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    save_plot(fig, "attrition_by_department.png", save)

    return fig


def plot_attrition_by_overtime(df, save=True):
    """Analyze the relationship between overtime and attrition."""

    data = df.copy()
    data["Attrition_Flag"] = (
        data["Attrition"] == "Yes"
    ).astype(int)

    rates = (
        data.groupby("OverTime")["Attrition_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="Attrition Rate")
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        data=rates,
        x="OverTime",
        y="Attrition Rate",
        hue="OverTime",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Attrition Rate by Overtime")
    ax.set_xlabel("Works Overtime")
    ax.set_ylabel("Employees Who Left (%)")

    fig.tight_layout()
    save_plot(fig, "attrition_by_overtime.png", save)

    return fig


def plot_attrition_by_age(df, save=True):
    """Explore attrition rates across employee age groups."""

    data = df.copy()

    data["Age Group"] = pd.cut(
        data["Age"],
        bins=[17, 25, 35, 45, 55, float("inf")],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56+"
        ]
    )

    data["Attrition_Flag"] = (
        data["Attrition"] == "Yes"
    ).astype(int)

    rates = (
        data.groupby("Age Group", observed=False)["Attrition_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="Attrition Rate")
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=rates,
        x="Age Group",
        y="Attrition Rate",
        hue="Age Group",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Attrition Rate by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Employees Who Left (%)")

    fig.tight_layout()
    save_plot(fig, "attrition_by_age_group.png", save)

    return fig


def plot_attrition_by_years_at_company(df, save=True):
    """Explore attrition rates based on years at the company."""

    data = df.copy()

    data["Company Tenure Group"] = pd.cut(
        data["YearsAtCompany"],
        bins=[-1, 1, 3, 5, 10, float("inf")],
        labels=[
            "0-1 years",
            "2-3 years",
            "4-5 years",
            "6-10 years",
            "Above 10 years"
        ]
    )

    data["Attrition_Flag"] = (
        data["Attrition"] == "Yes"
    ).astype(int)

    rates = (
        data.groupby(
            "Company Tenure Group",
            observed=False
        )["Attrition_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="Attrition Rate")
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=rates,
        x="Company Tenure Group",
        y="Attrition Rate",
        hue="Company Tenure Group",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Attrition Rate by Years at Company")
    ax.set_xlabel("Years at Company")
    ax.set_ylabel("Employees Who Left (%)")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    save_plot(fig, "attrition_by_years_at_company.png", save)

    return fig


def generate_eda_plots(df, categorical_cols, save=True):
    """Generate all exploratory data analysis visualizations."""

    figures = (
        plot_attrition_distribution(df, save),
        plot_correlation_heatmap(df, categorical_cols, save),
        plot_attrition_by_department(df, save),
        plot_attrition_by_overtime(df, save),
        plot_attrition_by_age(df, save),
        plot_attrition_by_years_at_company(df, save)
    )

    return figures
