from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)


# ==============================
# Configuration Dataclass
# ==============================
@dataclass
class DataQualityConfig:
    """Configuration for the Data Quality Engine."""

    # Required fields that must exist in the input dataframe
    required_cols: List[str] = field(default_factory=lambda: [
        "nric", "taxpayer_id", "postal_code",
        "annual_income_sgd", "total_reliefs_sgd", "chargeable_income_sgd"
    ])

    # Columns that should be coerced to numeric prior to accuracy checks
    numeric_cols: List[str] = field(default_factory=lambda: [
        "annual_income_sgd", "total_reliefs_sgd", "chargeable_income_sgd",
        "tax_payable_sgd", "tax_paid_sgd", "cpf_contributions_sgd", "foreign_income_sgd"
    ])

    # Optional identifier columns to keep in the details output (if present)
    id_cols: List[str] = field(default_factory=lambda: ["record_id", "nric", "taxpayer_id"])

    # Regex patterns (format validity)
    nric_regex: str = r"^[STFG]\d{7}[A-Z]$"
    postal_regex: str = r"^\d{6}$"

    # Accuracy tolerance (exact by default; set to >0 for floating tolerance)
    accuracy_tolerance: float = 0.0

    # --------------------------
    # Helpers for config loading
    # --------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataQualityConfig":
        """
        Create a config from a dict. Unknown keys are ignored.
        Accepts either top-level keys or nested under "engine".
        """
        engine = d.get("engine", d)
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in engine.items() if k in fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================
# Data Quality Engine (OOP)
# ==============================
class DataQualityEngine:
    """Runs data quality checks and produces summary, details, and field completeness dataframes."""

    def __init__(self, config: Optional[DataQualityConfig] = None) -> None:
        self.config = config or DataQualityConfig()

    # ---------- Public API ----------
    def build_reports(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return (dq_summary_df, dq_details_df, field_completeness_df)
        """
        self._assert_required_columns(df)
        n = len(df)
        df = df.copy()

        # Normalize numeric columns
        self._coerce_numeric(df, self.config.numeric_cols)

        # Compute flags and diagnostics
        completeness_flag = self._validate_completeness(df)
        validity_flag, validity_diags = self._validate_validity(df)
        accuracy_flag, accuracy_diags = self._validate_accuracy(df)

        # Attach flags for downstream computations
        df["completeness_flag"] = completeness_flag
        df["validity_flag"] = validity_flag
        df["accuracy_flag"] = accuracy_flag

        # Scores
        pct = (lambda x: round(float(x) / n * 100, 2) if n else 0.0)
        completeness_score = pct(completeness_flag.sum())
        validity_score = pct(validity_flag.sum())
        accuracy_score = pct(accuracy_flag.sum())
        overall_dq_score = round(
            np.mean([completeness_score, validity_score, accuracy_score]), 2
        ) if n else 0.0

        # Metric-level diagnostics
        missing_nric_count = int(df["nric"].isna().sum())
        missing_taxpayer_id_count = int(df["taxpayer_id"].isna().sum())
        invalid_nric_count = int((~validity_diags["nric_valid"]).sum())
        invalid_postal_code_count = int((~validity_diags["postal_valid"]).sum())
        accuracy_mismatch_count = int(accuracy_diags["mismatch_mask"].sum())
        accuracy_null_operands_count = int(accuracy_diags["null_operands_mask"].sum())

        # Build summary dataframe
        dq_summary_df = self._build_summary_df(
            n=n,
            completeness_flag=completeness_flag,
            validity_flag=validity_flag,
            accuracy_flag=accuracy_flag,
            completeness_score=completeness_score,
            validity_score=validity_score,
            accuracy_score=accuracy_score,
            overall_dq_score=overall_dq_score,
            missing_nric_count=missing_nric_count,
            missing_taxpayer_id_count=missing_taxpayer_id_count,
            invalid_nric_count=invalid_nric_count,
            invalid_postal_code_count=invalid_postal_code_count,
            accuracy_mismatch_count=accuracy_mismatch_count,
            accuracy_null_operands_count=accuracy_null_operands_count,
            pct=pct,
        )

        # Build row-level details with failure reasons
        dq_details_df = self._build_details_df(
            df=df,
            validity_diags=validity_diags,
            accuracy_diags=accuracy_diags,
        )

        # Build field-level completeness
        field_completeness_df = self._build_field_completeness_df(
            df=df, fields_for_completeness=self.config.required_cols, pct=pct
        )

        return dq_summary_df, dq_details_df, field_completeness_df

    # NEW: Public API to get a fully annotated dataframe with flags + reasons
    def annotate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of df with:
          - numeric coercion
          - completeness_flag, validity_flag, accuracy_flag
          - reason_* columns, failure_reasons
          - dq_pass
          - income/tax derived metrics (see _add_income_tax_derivatives)
        """
        self._assert_required_columns(df)
        df = df.copy()
        self._coerce_numeric(df, self.config.numeric_cols)

        # Flags
        completeness_flag = self._validate_completeness(df)
        validity_flag, _ = self._validate_validity(df)
        accuracy_flag, accuracy_diags = self._validate_accuracy(df)

        df["completeness_flag"] = completeness_flag
        df["validity_flag"] = validity_flag
        df["accuracy_flag"] = accuracy_flag

        # Reason flags
        reason_cols: Dict[str, pd.Series] = {}
        reason_cols["reason_missing_nric"] = df["nric"].isna()
        reason_cols["reason_missing_taxpayer_id"] = df["taxpayer_id"].isna()
        reason_cols["reason_missing_postal_code"] = df["postal_code"].isna()

        reason_cols["reason_invalid_postal_code_format"] = (
            ~reason_cols["reason_missing_postal_code"]
        ) & (~df["postal_code"].astype(str).str.match(self.config.postal_regex, na=False))

        reason_cols["reason_invalid_nric_format"] = (
            ~df["nric"].isna()
        ) & (~df["nric"].astype(str).str.match(self.config.nric_regex, na=False))

        reason_cols["reason_accuracy_null_operands"] = accuracy_diags["null_operands_mask"]
        reason_cols["reason_chargeable_income_mismatch"] = accuracy_diags["mismatch_mask"]

        for c, s in reason_cols.items():
            df[c] = s.fillna(False)

        # failure_reasons + dq_pass
        reason_label_map = {
            "reason_missing_nric": "missing_nric",
            "reason_missing_taxpayer_id": "missing_taxpayer_id",
            "reason_missing_postal_code": "missing_postal_code",
            "reason_invalid_postal_code_format": "invalid_postal_code_format",
            "reason_invalid_nric_format": "invalid_nric_format",
            "reason_accuracy_null_operands": "accuracy_null_operands",
            "reason_chargeable_income_mismatch": "chargeable_income_mismatch",
        }
        cols_in_order = list(reason_label_map.keys())

        df["failure_reasons"] = (
            df[cols_in_order]
              .apply(lambda row: "; ".join([reason_label_map[c] for c, v in row.items() if v]), axis=1)
              .replace("", np.nan)
        )
        df["dq_pass"] = df[["completeness_flag", "validity_flag", "accuracy_flag"]].all(axis=1)

        # Derived metrics for income/tax views
        df = self._add_income_tax_derivatives(df)
        return df

    # ---------- Internal: Validation ----------
    def _validate_completeness(self, df: pd.DataFrame) -> pd.Series:
        return df["nric"].notna() & df["taxpayer_id"].notna()

    def _validate_validity(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        nric_valid = df["nric"].astype(str).str.match(self.config.nric_regex, na=False)
        postal_valid = df["postal_code"].astype(str).str.match(self.config.postal_regex, na=False)
        validity_flag = nric_valid & postal_valid
        diagnostics = {
            "nric_valid": nric_valid,
            "postal_valid": postal_valid,
        }
        return validity_flag, diagnostics

    def _validate_accuracy(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        chargeable_income_sgd == annual_income_sgd - total_reliefs_sgd
        If accuracy_tolerance > 0, uses absolute difference <= tolerance.
        """
        expected = df["annual_income_sgd"] - df["total_reliefs_sgd"]
        null_operands_mask = df[["annual_income_sgd", "total_reliefs_sgd", "chargeable_income_sgd"]].isna().any(axis=1)

        if self.config.accuracy_tolerance > 0:
            diff = (df["chargeable_income_sgd"] - expected).abs()
            accuracy_flag = (~null_operands_mask) & (diff <= self.config.accuracy_tolerance)
            mismatch_mask = (~null_operands_mask) & (diff > self.config.accuracy_tolerance)
        else:
            accuracy_flag = (~null_operands_mask) & (df["chargeable_income_sgd"] == expected)
            mismatch_mask = (~null_operands_mask) & (df["chargeable_income_sgd"] != expected)

        diagnostics = {
            "expected_ci": expected,
            "null_operands_mask": null_operands_mask,
            "mismatch_mask": mismatch_mask,
        }
        return accuracy_flag, diagnostics

    # ---------- Internal: Builders ----------
    def _build_summary_df(
        self,
        n: int,
        completeness_flag: pd.Series,
        validity_flag: pd.Series,
        accuracy_flag: pd.Series,
        completeness_score: float,
        validity_score: float,
        accuracy_score: float,
        overall_dq_score: float,
        missing_nric_count: int,
        missing_taxpayer_id_count: int,
        invalid_nric_count: int,
        invalid_postal_code_count: int,
        accuracy_mismatch_count: int,
        accuracy_null_operands_count: int,
        pct,
    ) -> pd.DataFrame:
        summary_rows = [
            {
                "metric": "completeness",
                "rule": "nric & taxpayer_id not null",
                "total_records": n,
                "pass_count": int(completeness_flag.sum()),
                "fail_count": int((~completeness_flag).sum()),
                "pass_pct": completeness_score,
                "fail_pct": pct((~completeness_flag).sum()),
                "missing_nric_count": missing_nric_count,
                "missing_taxpayer_id_count": missing_taxpayer_id_count,
            },
            {
                "metric": "validity",
                "rule": "NRIC & postal_code match formats",
                "total_records": n,
                "pass_count": int(validity_flag.sum()),
                "fail_count": int((~validity_flag).sum()),
                "pass_pct": validity_score,
                "fail_pct": pct((~validity_flag).sum()),
                "invalid_nric_count": invalid_nric_count,
                "invalid_postal_code_count": invalid_postal_code_count,
            },
            {
                "metric": "accuracy",
                "rule": "chargeable_income_sgd == annual_income_sgd - total_reliefs_sgd",
                "total_records": n,
                "pass_count": int(accuracy_flag.sum()),
                "fail_count": int((~accuracy_flag).sum()),
                "pass_pct": accuracy_score,
                "fail_pct": pct((~accuracy_flag).sum()),
                "mismatched_values_count": accuracy_mismatch_count,
                "null_operands_count": accuracy_null_operands_count,
            },
        ]
        dq_summary_df = pd.DataFrame(summary_rows)

        overall_row = {
            "metric": "overall_score (avg of 3)",
            "rule": "",
            "total_records": n,
            "pass_count": "",
            "fail_count": "",
            "pass_pct": round(np.mean([r["pass_pct"] for r in summary_rows]), 2) if n else 0.0,
            "fail_pct": round(100 - (round(np.mean([r["pass_pct"] for r in summary_rows]), 2) if n else 0.0), 2) if n else 0.0,
        }
        dq_summary_df = pd.concat([dq_summary_df, pd.DataFrame([overall_row])], ignore_index=True)
        return dq_summary_df

    def _build_details_df(
        self,
        df: pd.DataFrame,
        validity_diags: Dict[str, pd.Series],
        accuracy_diags: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        # Reason flags
        reason_cols = {}
        reason_cols["reason_missing_nric"] = df["nric"].isna()
        reason_cols["reason_missing_taxpayer_id"] = df["taxpayer_id"].isna()
        reason_cols["reason_missing_postal_code"] = df["postal_code"].isna()

        # Format checks (mark invalid only when present)
        postal_regex = self.config.postal_regex
        nric_regex = self.config.nric_regex

        reason_cols["reason_invalid_postal_code_format"] = (
            ~reason_cols["reason_missing_postal_code"]
        ) & (~df["postal_code"].astype(str).str.match(postal_regex, na=False))

        reason_cols["reason_invalid_nric_format"] = (
            ~df["nric"].isna()
        ) & (~df["nric"].astype(str).str.match(nric_regex, na=False))

        # Accuracy diagnostics
        reason_cols["reason_accuracy_null_operands"] = accuracy_diags["null_operands_mask"]
        reason_cols["reason_chargeable_income_mismatch"] = accuracy_diags["mismatch_mask"]

        # Attach reason flags to df
        for c, s in reason_cols.items():
            df[c] = s.fillna(False)

        # Human-readable reason labels
        reason_label_map = {
            "reason_missing_nric": "missing_nric",
            "reason_missing_taxpayer_id": "missing_taxpayer_id",
            "reason_missing_postal_code": "missing_postal_code",
            "reason_invalid_postal_code_format": "invalid_postal_code_format",
            "reason_invalid_nric_format": "invalid_nric_format",
            "reason_accuracy_null_operands": "accuracy_null_operands",
            "reason_chargeable_income_mismatch": "chargeable_income_mismatch",
        }
        cols_in_order = list(reason_label_map.keys())

        df["failure_reasons"] = (
            df[cols_in_order]
            .apply(lambda row: "; ".join([reason_label_map[c] for c, v in row.items() if v]), axis=1)
            .replace("", np.nan)
        )

        # Overall DQ pass if all three flags pass
        df["dq_pass"] = df[["completeness_flag", "validity_flag", "accuracy_flag"]].all(axis=1)

        # Prepare details view
        id_cols_present = [c for c in self.config.id_cols if c in df.columns]
        keep_cols = id_cols_present + [
            "postal_code", "annual_income_sgd", "total_reliefs_sgd", "chargeable_income_sgd",
            "completeness_flag", "validity_flag", "accuracy_flag", "dq_pass", "failure_reasons"
        ] + cols_in_order

        return df[keep_cols].copy()

    def _build_field_completeness_df(
        self,
        df: pd.DataFrame,
        fields_for_completeness: List[str],
        pct,
    ) -> pd.DataFrame:
        n = len(df)
        if n == 0:
            return pd.DataFrame(columns=["field", "null_count", "null_pct", "non_null_count", "non_null_pct"])

        out = pd.DataFrame({
            "field": fields_for_completeness,
            "null_count": [int(df[c].isna().sum()) for c in fields_for_completeness],
        })
        out["null_pct"] = out["null_count"].apply(pct)
        out["non_null_count"] = n - out["null_count"]
        out["non_null_pct"] = 100 - out["null_pct"]
        return out

    # ---------- Internal: Utilities ----------
    def _assert_required_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.config.required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> None:
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # NEW: income/tax derived metrics to enrich themed views
    @staticmethod
    def _add_income_tax_derivatives(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Safe math with NaN handling
        df["income_after_reliefs_sgd"] = df["annual_income_sgd"] - df["total_reliefs_sgd"]
        df["tax_balance_sgd"] = df.get("tax_payable_sgd", np.nan) - df.get("tax_paid_sgd", np.nan)

        # Effective tax rate: tax_payable / chargeable_income
        denom = df["chargeable_income_sgd"].replace({0: np.nan})
        df["effective_tax_rate_pct"] = (df.get("tax_payable_sgd", np.nan) / denom) * 100

        # Relief ratio: total_reliefs / annual_income
        denom2 = df["annual_income_sgd"].replace({0: np.nan})
        df["relief_ratio_pct"] = (df["total_reliefs_sgd"] / denom2) * 100
        return df

    # NEW: Build themed views (returns dict of name -> DataFrame)
    def _build_themed_views(self, annotated_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create four themed DataFrames from the annotated_df.
        Only keep columns that actually exist in annotated_df (robust to missing fields).
        """
        def pick(cols: List[str]) -> List[str]:
            return [c for c in cols if c in annotated_df.columns]

        # 1) Taxpayer info (NRIC, demographics)
        taxpayer_info_cols = pick([
            "taxpayer_id", "nric", "full_name",
            "residential_status", "number_of_dependents", "housing_type",
            "assessment_year"
        ])
        taxpayer_info_df = annotated_df[taxpayer_info_cols].copy()

        # 2) Income and tax calculations (+ derived metrics)
        income_tax_cols = pick([
            "taxpayer_id", "nric",
            "annual_income_sgd", "total_reliefs_sgd", "chargeable_income_sgd",
            "cpf_contributions_sgd", "foreign_income_sgd",
            "tax_payable_sgd", "tax_paid_sgd",
            "income_after_reliefs_sgd", "tax_balance_sgd",
            "effective_tax_rate_pct", "relief_ratio_pct"
        ])
        income_tax_df = annotated_df[income_tax_cols].copy()

        # 3) Filing details and compliance data (include DQ flags)
        filing_compliance_cols = pick([
            "taxpayer_id", "nric",
            "filing_status", "assessment_year", "filing_date",
            "completeness_flag", "validity_flag", "accuracy_flag", "dq_pass", "failure_reasons"
        ])
        filing_compliance_df = annotated_df[filing_compliance_cols].copy()

        # 4) Geographic and occupation classifications
        geo_occupation_cols = pick([
            "taxpayer_id", "nric",
            "postal_code", "housing_type", "occupation", "residential_status"
        ])
        geo_occupation_df = annotated_df[geo_occupation_cols].copy()

        return {
            "taxpayer_info": taxpayer_info_df,
            "income_tax": income_tax_df,
            "filing_compliance": filing_compliance_df,
            "geo_occupation": geo_occupation_df,
        }


# ==============================
# Runner helpers (config-driven)
# ==============================
def _expand_path(p: str) -> Path:
    """Expand ~ and env vars, accept forward or backslashes."""
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


def load_json_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = _expand_path(str(config_path))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level JSON must be an object/dict.")
    return cfg


def run_with_config(config_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    High-level runner:
      - reads JSON config
      - loads CSV
      - builds engine from config
      - runs reports
      - saves outputs (Excel/CSVs) if configured
      - builds and saves themed CSVs if configured
    """
    cfg = load_json_config(config_path)

    # 1) Input settings
    input_cfg = cfg.get("input", cfg)
    file_path = input_cfg.get("file_path")
    if not file_path:
        raise ValueError("`file_path` is required in the config.")
    file_path = _expand_path(file_path)

    read_csv_kwargs = input_cfg.get("read_csv_kwargs", {})
    if not isinstance(read_csv_kwargs, dict):
        raise ValueError("`read_csv_kwargs` must be a JSON object/dict if provided.")

    # 2) Engine settings
    dq_conf = DataQualityConfig.from_dict(cfg)

    # 3) Load data
    df = pd.read_csv(file_path, **read_csv_kwargs)

    # 4) Build reports
    engine = DataQualityEngine(dq_conf)
    summary_df, details_df, field_comp_df = engine.build_reports(df)

    # 5) Output settings (Excel + CSVs)
    output_cfg = cfg.get("output", {})
    excel_path = output_cfg.get("excel_path")
    if excel_path:
        excel_path = _expand_path(excel_path)
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="summary")
            details_df.to_excel(writer, index=False, sheet_name="details")
            field_comp_df.to_excel(writer, index=False, sheet_name="field_completeness")

    details_csv = output_cfg.get("details_csv")
    summary_csv = output_cfg.get("summary_csv")
    field_csv = output_cfg.get("field_completeness_csv")

    if summary_csv:
        p = _expand_path(summary_csv); p.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(p, index=False)
    if details_csv:
        p = _expand_path(details_csv); p.parent.mkdir(parents=True, exist_ok=True)
        details_df.to_csv(p, index=False)
    if field_csv:
        p = _expand_path(field_csv); p.parent.mkdir(parents=True, exist_ok=True)
        field_comp_df.to_csv(p, index=False)

    # 6) Themed CSV outputs (NEW)
    themed_cfg = output_cfg.get("views", {})
    if isinstance(themed_cfg, dict) and themed_cfg:
        # Use annotated df (flags + reasons + derived metrics)
        annotated_df = engine.annotate_df(df)

        themed_views = engine._build_themed_views(annotated_df)
        # Known keys the config can provide (each is optional)
        path_map = {
            "taxpayer_info": themed_cfg.get("taxpayer_info_csv"),
            "income_tax": themed_cfg.get("income_tax_csv"),
            "filing_compliance": themed_cfg.get("filing_compliance_csv"),
            "geo_occupation": themed_cfg.get("geo_occupation_csv"),
        }
        for name, path in path_map.items():
            if path:
                p = _expand_path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                themed_views[name].to_csv(p, index=False)

    return summary_df, details_df, field_comp_df


# ==============================
# CLI
# ==============================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Data Quality checks from a JSON configuration file."
    )
    p.add_argument(
        "-c", "--config",
        required=True,
        help="Path to JSON config file (e.g., config.json)"
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    summary_df, details_df, field_comp_df = run_with_config(args.config)

    # Brief console output
    print("\n=== SUMMARY (first 10 rows) ===")
    print(summary_df.head(10).to_string(index=False))
    print("\n=== DETAILS (first 10 rows) ===")
    print(details_df.head(10).to_string(index=False))
    print("\n=== FIELD COMPLETENESS (first 10 rows) ===")
    print(field_comp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()