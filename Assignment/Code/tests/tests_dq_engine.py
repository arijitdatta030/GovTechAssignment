# tests/test_dq_engine.py

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import from the module under test
# Make sure you run pytest from the project root where dq_engine.py lives
from dq_engine import (
    DataQualityConfig,
    DataQualityEngine,
    load_json_config,
    run_with_config,
)


# ----------------------------
# Fixtures & Sample Data
# ----------------------------
@pytest.fixture
def engine():
    """Default DataQualityEngine with exact equality (no tolerance)."""
    return DataQualityEngine(DataQualityConfig(accuracy_tolerance=0.0))


@pytest.fixture
def sample_df():
    """
    Five rows covering representative cases:

    r1: all valid -> all flags True
    r2: missing NRIC -> completeness False, validity False; accuracy True
    r3: invalid NRIC -> validity False; accuracy True
    r4: invalid postal + accuracy mismatch -> validity False, accuracy False
    r5: numeric strings + chargeable NaN -> accuracy False due to null operands
    """
    return pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5],
        "taxpayer_id": ["TP1", "TP2", "TP3", "TP4", "TP5"],
        "nric": ["S1234567D", np.nan, "X2345678A", "F7654321Z", "T0000000A"],
        "full_name": ["Alice", "Bob", "Charlie", "Dana", "Evan"],
        "filing_status": ["Filed", "Filed", "Not Filed", "Filed", "Filed"],
        "assessment_year": [2023, 2023, 2023, 2023, 2023],
        "filing_date": ["2023-03-15", "2023-03-20", None, "2023-03-18", "2023-03-22"],
        "annual_income_sgd": [100000, 90000, 50000, 120000, "100000"],
        "chargeable_income_sgd": [80000, 80000, 45000, 95000, np.nan],
        "tax_payable_sgd": [10000, 9000, 4000, 15000, 12000],
        "tax_paid_sgd": [8000, 9000, 3500, 15000, 12000],
        "total_reliefs_sgd": [20000, 10000, 5000, 20000, "20000"],
        "number_of_dependents": [1, 2, 0, 3, 0],
        "occupation": ["Engineer", "Analyst", "Teacher", "Nurse", "Designer"],
        "residential_status": ["Citizen", "PR", "Citizen", "Citizen", "PR"],
        "postal_code": ["123456", "234567", "654321", "12345", "345678"],
        "housing_type": ["HDB", "Condo", "HDB", "Landed", "HDB"],
        "cpf_contributions_sgd": [0, 0, 0, 0, 0],
        "foreign_income_sgd": [0, 0, 0, 0, 0],
    })


# ----------------------------
# Structural / Error Handling
# ----------------------------
def test_missing_required_columns_raises(engine):
    df = pd.DataFrame({"nric": ["S1234567D"], "taxpayer_id": ["TP1"]})
    with pytest.raises(ValueError) as exc:
        engine.build_reports(df)
    assert "Missing required columns" in str(exc.value)


def test_empty_dataframe_returns_empty_reports(engine):
    df = pd.DataFrame(columns=[
        "nric", "taxpayer_id", "postal_code",
        "annual_income_sgd", "total_reliefs_sgd", "chargeable_income_sgd",
    ])
    summary, details, field_comp = engine.build_reports(df)

    # summary -> 4 rows (3 metrics + overall), totals should be zero-ish
    assert len(summary) == 4
    assert summary["total_records"].fillna(0).eq(0).all()
    assert summary["pass_pct"].fillna(0).eq(0).all()

    # details & field completeness -> empty
    assert details.empty
    assert field_comp.empty


def test_config_from_dict_flat_and_nested():
    flat = {
        "required_cols": ["nric", "taxpayer_id", "postal_code"],
        "accuracy_tolerance": 0.01,
    }
    nested = {"engine": flat}

    cfg1 = DataQualityConfig.from_dict(flat)
    cfg2 = DataQualityConfig.from_dict(nested)

    assert cfg1.accuracy_tolerance == 0.01
    assert cfg2.required_cols == ["nric", "taxpayer_id", "postal_code"]


# ----------------------------
# Flags & Metrics
# ----------------------------
def test_completeness_flag(engine, sample_df):
    _, details, _ = engine.build_reports(sample_df)
    details = details.sort_values("record_id")
    assert details["completeness_flag"].tolist() == [True, False, True, True, True]


def test_validity_flag(engine, sample_df):
    _, details, _ = engine.build_reports(sample_df)
    details = details.sort_values("record_id")
    # r1 True, r2 False (missing NRIC), r3 False (invalid NRIC), r4 False (invalid postal), r5 True
    assert details["validity_flag"].tolist() == [True, False, False, False, True]


def test_accuracy_flag(engine, sample_df):
    _, details, _ = engine.build_reports(sample_df)
    details = details.sort_values("record_id")
    # r1 True, r2 True, r3 True, r4 False (mismatch), r5 False (null operands)
    assert details["accuracy_flag"].tolist() == [True, True, True, False, False]


def test_dq_pass_requires_all_three(engine, sample_df):
    _, details, _ = engine.build_reports(sample_df)
    details = details.sort_values("record_id")
    # Only r1 passes all three
    assert details["dq_pass"].tolist() == [True, False, False, False, False]


def test_summary_counts_and_overall_score(engine, sample_df):
    summary, _, _ = engine.build_reports(sample_df)

    s_comp = summary[summary["metric"] == "completeness"].iloc[0]
    s_valid = summary[summary["metric"] == "validity"].iloc[0]
    s_acc = summary[summary["metric"] == "accuracy"].iloc[0]
    s_overall = summary[summary["metric"].str.startswith("overall_score")].iloc[0]

    assert s_comp["total_records"] == 5
    assert s_comp["pass_count"] == 4     # 4/5 completeness pass
    assert s_comp["pass_pct"] == 80.0

    assert s_valid["pass_count"] == 2    # 2/5 validity pass
    assert s_valid["pass_pct"] == 40.0

    assert s_acc["pass_count"] == 3      # 3/5 accuracy pass
    assert s_acc["pass_pct"] == 60.0

    # overall = avg(80, 40, 60) = 60.0
    assert float(s_overall["pass_pct"]) == 60.0


def test_failure_reasons(engine, sample_df):
    _, details, _ = engine.build_reports(sample_df)
    details = details.set_index("record_id")

    # r1: none
    assert pd.isna(details.loc[1, "failure_reasons"])

    # r2: missing NRIC
    r2 = details.loc[2, "failure_reasons"]
    assert "missing_nric" in r2
    assert "invalid_nric_format" not in r2  # we only mark invalid format when present

    # r3: invalid NRIC format
    assert "invalid_nric_format" in details.loc[3, "failure_reasons"]

    # r4: invalid postal + mismatch
    r4 = details.loc[4, "failure_reasons"]
    assert "invalid_postal_code_format" in r4
    assert "chargeable_income_mismatch" in r4

    # r5: accuracy null operands
    assert "accuracy_null_operands" in details.loc[5, "failure_reasons"]


def test_field_completeness(engine, sample_df):
    _, _, field_comp = engine.build_reports(sample_df)
    s = dict(zip(field_comp["field"], field_comp["null_count"]))

    assert s["nric"] == 1
    assert s["taxpayer_id"] == 0
    assert s["postal_code"] == 0
    assert s["annual_income_sgd"] == 0
    assert s["total_reliefs_sgd"] == 0
    assert s["chargeable_income_sgd"] == 1


# ----------------------------
# annotate_df & derived metrics
# ----------------------------
def test_annotate_df_derivatives(engine, sample_df):
    annotated = engine.annotate_df(sample_df)
    r1 = annotated.loc[annotated["record_id"] == 1].iloc[0]

    # Derived
    assert float(r1["income_after_reliefs_sgd"]) == 80000.0
    assert float(r1["tax_balance_sgd"]) == 2000.0
    assert pytest.approx(float(r1["effective_tax_rate_pct"]), rel=1e-6) == 12.5
    assert pytest.approx(float(r1["relief_ratio_pct"]), rel=1e-6) == 20.0

    # Flags
    assert bool(r1["completeness_flag"]) is True
    assert bool(r1["validity_flag"]) is True
    assert bool(r1["accuracy_flag"]) is True
    assert bool(r1["dq_pass"]) is True


def test_themed_views(engine, sample_df):
    annotated = engine.annotate_df(sample_df)
    views = engine._build_themed_views(annotated)

    # Ensure the four keys exist
    for key in ["taxpayer_info", "income_tax", "filing_compliance", "geo_occupation"]:
        assert key in views
        assert isinstance(views[key], pd.DataFrame)

    # Check some columns presence
    assert set(["taxpayer_id", "nric"]).issubset(views["taxpayer_info"].columns)
    assert set(["income_after_reliefs_sgd", "effective_tax_rate_pct"]).issubset(views["income_tax"].columns)
    assert set(["filing_status", "dq_pass"]).issubset(views["filing_compliance"].columns)
    assert set(["postal_code", "occupation"]).issubset(views["geo_occupation"].columns)


# ----------------------------
# Runner / Config I/O
# ----------------------------
def test_load_json_config_roundtrip(tmp_path: Path):
    cfg_data = {"input": {"file_path": "dummy.csv"}}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

    loaded = load_json_config(cfg_path)
    assert loaded["input"]["file_path"] == "dummy.csv"


def test_run_with_config_csv_outputs_only(tmp_path: Path):
    # 1) Create a tiny CSV input
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({
        "nric": ["S1234567D"],
        "taxpayer_id": ["TPX"],
        "postal_code": ["123456"],
        "annual_income_sgd": [100000],
        "total_reliefs_sgd": [20000],
        "chargeable_income_sgd": [80000],
        "tax_payable_sgd": [12000],
        "tax_paid_sgd": [11000],
    }).to_csv(csv_path, index=False)

    # 2) Build a config that writes only CSVs (no Excel to avoid openpyxl dependency here)
    out_dir = tmp_path / "out"
    cfg = {
        "input": {
            "file_path": str(csv_path)
        },
        "output": {
            "summary_csv": str(out_dir / "summary.csv"),
            "details_csv": str(out_dir / "details.csv"),
            "field_completeness_csv": str(out_dir / "field_comp.csv"),
            "views": {
                "taxpayer_info_csv": str(out_dir / "views" / "taxpayer_info.csv"),
                "income_tax_csv": str(out_dir / "views" / "income_tax.csv"),
                "filing_compliance_csv": str(out_dir / "views" / "filing_compliance.csv"),
                "geo_occupation_csv": str(out_dir / "views" / "geo_occupation.csv")
            }
        }
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # 3) Run
    summary_df, details_df, field_comp_df = run_with_config(cfg_path)

    # 4) Assertions on returned DFs
    assert isinstance(summary_df, pd.DataFrame)
    assert isinstance(details_df, pd.DataFrame)
    assert isinstance(field_comp_df, pd.DataFrame)
    assert not summary_df.empty
    assert not details_df.empty
    assert not field_comp_df.empty

    # 5) Files should exist
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "details.csv").exists()
    assert (out_dir / "field_comp.csv").exists()
    assert (out_dir / "views" / "taxpayer_info.csv").exists()
    assert (out_dir / "views" / "income_tax.csv").exists()
    assert (out_dir / "views" / "filing_compliance.csv").exists()
    assert (out_dir / "views" / "geo_occupation.csv").exists()