"""
===============================================================================
 Enterprise Weekly Performance Detector — v2.0
===============================================================================
 Multi-layered anomaly detection, causal decomposition, leading indicators,
 and CEO-level executive narrative for fundraising performance monitoring.

 Key Differentiators from v1:
   1. Multi-layer anomaly detection (Z-score + IQR + CUSUM + Bayesian)
   2. Causal chain decomposition (volume vs yield vs mix)
   3. Leading indicator early-warning engine
   4. Seasonality-aware baselines (YoY comparison, holiday adjustment)
   5. Funnel-stage attribution (which stage is leaking)
   6. Financial impact estimation ($ at risk / $ opportunity)
   7. Executive narrative engine with actionable recommendations
   8. Data quality gate with pre-validation
   9. Config-driven parameter management
  10. Output lineage tracking (provenance metadata)

 Usage:
   from snowflake.snowpark.context import get_active_session
   session = get_active_session()
   detector = EnterpriseAnomalyDetector(session)
   results = detector.run()
===============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import logging
import json
import warnings
from collections import defaultdict
from functools import lru_cache

warnings.filterwarnings("ignore")

# =========================================================================
# 0. Logging & Configuration
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EnterpriseDetector")


class AlertSeverity(Enum):
    """CEO-level severity classification with action-orientation."""
    CRITICAL = "CRITICAL"       # Requires immediate CEO/CFO attention
    HIGH = "HIGH"               # Requires functional leader attention this week
    MEDIUM = "MEDIUM"           # Monitor closely; investigate in normal cadence
    LOW = "LOW"                 # Informational; normal variance
    INFO = "INFO"               # Positive signal or no issue


class AnomalyDirection(Enum):
    DROP = "DROP"
    SPIKE = "SPIKE"
    NORMAL = "NORMAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class DetectorConfig:
    """Central configuration — enterprise-grade parameter management.
    
    All tunable parameters live here. Can be loaded from YAML, env vars,
    or a database config table in production.
    """
    # --- Data Scope ---
    database: str = "ANALYTICS"
    schema: str = "CORE"
    lookback_weeks: int = 26           # How far back to fetch raw data
    baseline_weeks: int = 8            # Rolling baseline for anomaly detection
    min_baseline_obs: int = 4          # Minimum observations for statistical tests
    
    # --- Source Tables ---
    tables: List[str] = field(default_factory=lambda: [
        "FUNDRAISER_TRANSACTIONS",
        "FUNDRAISER_NET_TRANSACTIONS",
        "DONOR_FUNNEL",
        "LANDING_PAGE_TO_DONOR_FUNNEL",
        "CO_FUNNEL",
        "LANDING_PAGE_TO_CO_FUNNEL",
        "NET_KPI_TABLES_WEEKLY",
        "GOOGLE_ADS_STATS",
        "SHARE_INTENT",
        "FUNDRAISER_VIEWS",
        # "WEEKLY_CRISIS_RANKING_GDV",  -- uncomment if this table exists in your schema
    ])
    
    # --- Anomaly Thresholds ---
    # Layer 1: Percentage change
    pct_drop_threshold: float = -10.0
    pct_spike_threshold: float = 10.0
    pct_critical_drop: float = -25.0    # Triggers CRITICAL severity
    
    # Layer 2: Z-score
    z_score_threshold: float = 2.0       # |z| > this = unusual
    z_score_critical: float = 3.5        # |z| > this = critical
    
    # Layer 3: IQR-based (robust to outliers in baseline)
    iqr_multiplier: float = 1.5          # Standard Tukey fences
    
    # Layer 4: CUSUM (for persistent drift detection)
    cusum_threshold: float = 3.0         # CUSUM decision interval
    cusum_drift: float = 0.5             # Allowable slack
    
    # Layer 5: Bayesian probability (P(real change | data))
    bayesian_posterior_threshold: float = 0.80
    
    # --- Seasonality ---
    enable_yoy_comparison: bool = True
    yoy_lookback_weeks: int = 52
    
    # --- Leading Indicators ---
    enable_leading_indicators: bool = True
    leading_indicator_lags: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    # --- Output ---
    output_prefix: str = "AI_ENTERPRISE"
    save_intermediate: bool = True
    
    # --- Financial Impact ---
    annual_target_gdv: float = 1_000_000_000  # $1B annual target (adjust per context)
    weekly_impact_floor: float = 10_000       # Don't flag <$10K movements

    # --- Data Quality ---
    max_null_fraction: float = 0.3            # Fail if >30% nulls in metric columns
    min_total_rows: int = 10                  # Fail if <10 rows per table


# =========================================================================
# 1. Schema Discovery & Profiling
# =========================================================================

class SchemaDiscovery:
    """Discovers and profiles Snowflake table schemas with enterprise metadata."""

    def __init__(self, session, config: DetectorConfig):
        self.session = session
        self.config = config
        self.columns_df = None
        self.table_profile = None

    def discover(self) -> pd.DataFrame:
        """Fetch all column metadata for configured tables."""
        logger.info(f"Discovering schema for {len(self.config.tables)} tables in "
                     f"{self.config.database}.{self.config.schema}")
        
        tables_list = ",".join([f"'{t}'" for t in self.config.tables])
        
        self.columns_df = self.session.sql(f"""
            SELECT
                table_name,
                column_name,
                data_type,
                ordinal_position,
                is_nullable,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM {self.config.database}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = '{self.config.schema}'
              AND table_name IN ({tables_list})
            ORDER BY table_name, ordinal_position
        """).to_pandas()
        
        self.columns_df.columns = [c.upper() for c in self.columns_df.columns]
        logger.info(f"Discovered {len(self.columns_df)} columns across "
                     f"{self.columns_df['TABLE_NAME'].nunique()} tables")
        return self.columns_df

    def build_profile(self) -> pd.DataFrame:
        """Build a semantic profile of each table: what columns serve which role."""
        if self.columns_df is None:
            self.discover()

        def table_cols(table):
            return self.columns_df[self.columns_df["TABLE_NAME"] == table]["COLUMN_NAME"].tolist()

        def pick_col(table, keywords):
            cols = table_cols(table)
            for kw in keywords:
                matches = [c for c in cols if kw.lower() in c.lower()]
                if matches:
                    return matches[0]
            return None

        profile = []
        for table in self.config.tables:
            profile.append({
                "table_name": table,
                "date_col": self._pick_date_col(table),
                "fundraiser_col": pick_col(table, ["fundraiser_id", "views_fundraiser_id", "fundraiser"]),
                "country_col": pick_col(table, ["fundraiser_country_iso", "views_country_code",
                                                  "landing_page_country_code", "campaign_country",
                                                  "country_code", "country"]),
                "device_col": pick_col(table, ["device_name", "device_type", "device_udid", "device"]),
                "source_col": pick_col(table, ["record_source", "platform_source", "utm_source",
                                                "campaign_source", "ea_source", "source", "channel"]),
                "gdv_col": pick_col(table, ["net_gdv", "gross_gdv", "total_donation_amount",
                                             "donation_amount", "amount_raised", "goal_amount",
                                             "gdv", "amount"]),
                "nac2_col": pick_col(table, ["nac2_count", "is_nac2", "nac2"]),
                "complete_col": pick_col(table, ["complete_users", "complete", "purchase",
                                                  "num_donations", "donation_id"]),
                "view_col": pick_col(table, ["page_view", "is_fundraiser_page_view",
                                              "view_event", "views_event_id", "view"]),
                "click_col": pick_col(table, ["clicks", "donate_click", "click"]),
                "cost_col": pick_col(table, ["cost", "spend"]),
            })
        
        self.table_profile = pd.DataFrame(profile)
        logger.info(f"Built profile for {len(self.table_profile)} tables")
        return self.table_profile

    def _pick_date_col(self, table):
        """Intelligently pick the best date/timestamp column for a table."""
        table_df = self.columns_df[self.columns_df["TABLE_NAME"] == table].copy()
        if table_df.empty:
            return None
        
        date_candidates = table_df[
            table_df["DATA_TYPE"].apply(lambda dt: self._is_date_type(str(dt)))
        ]
        
        preferred_keywords = [
            "week_start", "report_week", "week",
            "event_date", "event_timestamp", "payment_created",
            "created", "timestamp", "date",
            "transaction_at", "report_date"
        ]
        
        for kw in preferred_keywords:
            matches = date_candidates[
                date_candidates["COLUMN_NAME"].str.contains(kw, case=False, na=False)
            ]
            if not matches.empty:
                return matches.iloc[0]["COLUMN_NAME"]
        
        if not date_candidates.empty:
            return date_candidates.iloc[0]["COLUMN_NAME"]
        return None

    @staticmethod
    def _is_numeric_type(dt: str) -> bool:
        dt = dt.upper()
        return any(x in dt for x in ["NUMBER", "DECIMAL", "NUMERIC", "INT", "FLOAT", "DOUBLE", "REAL"])

    @staticmethod
    def _is_date_type(dt: str) -> bool:
        dt = dt.upper()
        return any(x in dt for x in ["DATE", "TIMESTAMP"])


# =========================================================================
# 2. Data Quality Gate
# =========================================================================

class DataQualityGate:
    """Pre-execution validation that data is fit for analysis.
    
    Checks:
      - Row count sufficiency
      - Null fraction in key columns
      - Date range completeness
      - Metric column type consistency
    """

    def __init__(self, session, config: DetectorConfig, schema: SchemaDiscovery):
        self.session = session
        self.config = config
        self.schema = schema
        self.issues: List[str] = []

    def validate(self) -> bool:
        """Run all quality checks. Returns True if passable, False if fatal."""
        logger.info("Running data quality gate...")
        self.issues = []
        
        if self.schema.columns_df is None:
            self.issues.append("Schema not discovered — cannot validate")
            return False

        # 1. Table existence & row counts
        self._check_table_row_counts()
        
        # 2. Null fraction in metric columns
        self._check_null_fractions()
        
        # 3. Date range completeness
        self._check_date_coverage()
        
        # 4. Column type consistency for metric aggregations
        self._check_metric_column_types()
        
        if self.issues:
            logger.warning(f"Data quality issues found ({len(self.issues)}):")
            for issue in self.issues:
                logger.warning(f"  ⚠  {issue}")
            # Return True if no FATAL issues — let downstream handle warnings
            fatal = [i for i in self.issues if i.startswith("FATAL")]
            if fatal:
                logger.error("FATAL quality issues — aborting analysis")
                return False
            return True
        
        logger.info("✅ Data quality gate passed — all checks OK")
        return True

    def _check_table_row_counts(self):
        """Ensure tables have sufficient data. Missing tables are skipped with a warning (non-fatal)."""
        existing_tables = []
        for table in list(self.config.tables):  # copy list so we can mutate if needed
            try:
                result = self.session.sql(
                    f"SELECT COUNT(*) AS cnt FROM {self.config.database}.{self.config.schema}.{table}"
                ).collect()[0]
                count = result["CNT"]
                if count < self.config.min_total_rows:
                    self.issues.append(
                        f"Table {table} has only {count} rows (min: {self.config.min_total_rows})"
                    )
                existing_tables.append(table)
            except Exception as e:
                # Table doesn't exist or isn't accessible — skip it, don't abort
                self.issues.append(f"Table {table} does not exist or is not accessible — skipping")

    def _check_null_fractions(self):
        """Check null fraction in key date and metric columns."""
        for _, row in self.schema.table_profile.iterrows():
            table = row["table_name"]
            key_cols = [c for c in [row["date_col"], row["gdv_col"]] if c]
            for col in key_cols:
                try:
                    result = self.session.sql(f"""
                        SELECT COUNT(*) AS total,
                               SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS nulls
                        FROM {self.config.database}.{self.config.schema}.{table}
                    """).collect()[0]
                    null_frac = result["NULLS"] / max(result["TOTAL"], 1)
                    if null_frac > self.config.max_null_fraction:
                        self.issues.append(
                            f"Table {table}.{col} is {null_frac:.0%} null "
                            f"(max: {self.config.max_null_fraction:.0%})"
                        )
                except Exception as e:
                    self.issues.append(f"Cannot check nulls for {table}.{col}: {e}")

    def _check_date_coverage(self):
        """Ensure date columns span the expected lookback period."""
        for _, row in self.schema.table_profile.iterrows():
            date_col = row["date_col"]
            if not date_col:
                continue
            try:
                result = self.session.sql(f"""
                    SELECT MIN({date_col}) AS min_dt, MAX({date_col}) AS max_dt,
                           DATEDIFF('day', MIN({date_col}), MAX({date_col})) AS day_span
                    FROM {self.config.database}.{self.config.schema}.{row['table_name']}
                """).collect()[0]
                min_dt, max_dt = result["MIN_DT"], result["MAX_DT"]
                if max_dt and min_dt:
                    span_weeks = result["DAY_SPAN"] / 7.0
                    if span_weeks < self.config.lookback_weeks * 0.5:
                        self.issues.append(
                            f"Table {row['table_name']}.{date_col} spans only "
                            f"{span_weeks:.0f} weeks (expected ~{self.config.lookback_weeks})"
                        )
            except Exception as e:
                self.issues.append(f"Cannot check date coverage for {row['table_name']}: {e}")

    def _check_metric_column_types(self):
        """Flag non-numeric columns mapped as metric columns."""
        for _, row in self.schema.table_profile.iterrows():
            table = row["table_name"]
            for role, col in [("gdv", row["gdv_col"]), ("cost", row["cost_col"])]:
                if not col:
                    continue
                dtype_row = self.schema.columns_df[
                    (self.schema.columns_df["TABLE_NAME"] == table) &
                    (self.schema.columns_df["COLUMN_NAME"] == col)
                ]
                if not dtype_row.empty:
                    dt = str(dtype_row.iloc[0]["DATA_TYPE"]).upper()
                    if not SchemaDiscovery._is_numeric_type(dt):
                        self.issues.append(
                            f"Table {table}.{col} ({role}) has type {dt} — "
                            f"expected numeric; will use TRY_TO_NUMBER"
                        )


# =========================================================================
# 3. Weekly Data Ingestion
# =========================================================================

class WeeklyDataIngestor:
    """Builds and executes weekly aggregate SQL with enterprise error handling."""

    def __init__(self, session, config: DetectorConfig, schema: SchemaDiscovery):
        self.session = session
        self.config = config
        self.schema = schema

    def ingest_all(self) -> pd.DataFrame:
        """Pull weekly metrics for all tables, return a unified long-format DataFrame."""
        logger.info("Ingesting weekly data from all sources...")
        
        queries = []
        for _, row in self.schema.table_profile.iterrows():
            q = self._build_weekly_query(row)
            if q:
                queries.append(q)
        
        if not queries:
            raise ValueError("No valid queries could be built — check table profiles")
        
        union_sql = "\nUNION ALL\n".join(queries)
        
        logger.debug(f"Executing union query across {len(queries)} tables")
        result = self.session.sql(union_sql).to_pandas()
        result.columns = [c.upper() for c in result.columns]
        result["WEEK_START_DATE"] = pd.to_datetime(result["WEEK_START_DATE"])
        
        logger.info(f"Ingested {len(result)} rows across "
                     f"{result['SOURCE_TABLE'].nunique()} tables, "
                     f"{result['WEEK_START_DATE'].nunique()} weeks")
        return result

    def _build_weekly_query(self, row) -> Optional[str]:
        """Build a single weekly aggregation query for one table."""
        table = row["table_name"]
        date_col = row["date_col"]
        
        if not date_col:
            logger.warning(f"Skipping {table}: no date column found")
            return None

        date_expr = self._safe_date_expr(table, date_col)
        
        select_parts = [
            f"'{table}' AS source_table",
            f"DATE_TRUNC('WEEK', {date_expr})::DATE AS week_start_date"
        ]
        group_parts = ["1", "2"]

        # Dimension columns (with NULL fallback)
        for dim_col, dim_name in [
            (row["country_col"], "country"),
            (row["device_col"], "device"),
            (row["source_col"], "traffic_source"),
        ]:
            if dim_col:
                select_parts.append(f"{dim_col}::VARCHAR AS {dim_name}")
                group_parts.append(str(len(select_parts)))
            else:
                select_parts.append(f"NULL::VARCHAR AS {dim_name}")

        # Metric expressions
        metric_parts = []
        metric_parts.append(self._metric_count_distinct(row["fundraiser_col"], "fundraisers"))
        metric_parts.append(self._metric_sum(table, row["gdv_col"], "gdv"))
        
        # NAC2: count non-null (date/boolean)
        if row["nac2_col"]:
            metric_parts.append(f"COUNT_IF({row['nac2_col']} IS NOT NULL) AS nac2")
        else:
            metric_parts.append("NULL AS nac2")
        
        # Completes: count non-null
        if row["complete_col"]:
            metric_parts.append(f"COUNT_IF({row['complete_col']} IS NOT NULL) AS completes")
        else:
            metric_parts.append("NULL AS completes")
        
        # Views and Clicks: numeric SUM or count non-null
        for metric_col, alias in [(row["view_col"], "views"), (row["click_col"], "clicks")]:
            if metric_col:
                dtype = self._col_type(table, metric_col)
                if dtype and SchemaDiscovery._is_numeric_type(dtype):
                    metric_parts.append(f"SUM({metric_col}) AS {alias}")
                else:
                    metric_parts.append(f"COUNT_IF({metric_col} IS NOT NULL) AS {alias}")
            else:
                metric_parts.append(f"NULL AS {alias}")
        
        metric_parts.append(self._metric_sum(table, row["cost_col"], "cost"))

        start_date_sql = f"DATEADD(WEEK, -{self.config.lookback_weeks}, CURRENT_DATE())"
        end_date_sql = "CURRENT_DATE()"

        return f"""
            SELECT {', '.join(select_parts)}, {', '.join(metric_parts)}
            FROM {self.config.database}.{self.config.schema}.{table}
            WHERE {date_expr} >= {start_date_sql}
              AND {date_expr} < {end_date_sql}
              AND {date_expr} IS NOT NULL
            GROUP BY {', '.join(group_parts)}
        """

    def _safe_date_expr(self, table, column):
        dtype = self._col_type(table, column)
        if dtype and SchemaDiscovery._is_date_type(dtype):
            return f"{column}::TIMESTAMP_NTZ"
        return f"TRY_TO_TIMESTAMP_NTZ({column}::VARCHAR)"

    def _col_type(self, table, column):
        if not column:
            return None
        match = self.schema.columns_df[
            (self.schema.columns_df["TABLE_NAME"] == table) &
            (self.schema.columns_df["COLUMN_NAME"] == column)
        ]
        if match.empty:
            return None
        return match.iloc[0]["DATA_TYPE"]

    def _metric_sum(self, table, column, alias):
        if not column:
            return f"NULL AS {alias}"
        dtype = self._col_type(table, column)
        if dtype and SchemaDiscovery._is_numeric_type(dtype):
            return f"SUM({column}) AS {alias}"
        return f"SUM(TRY_TO_NUMBER({column}::VARCHAR)) AS {alias}"

    def _metric_count_distinct(self, column, alias):
        if not column:
            return f"NULL AS {alias}"
        return f"COUNT(DISTINCT {column}) AS {alias}"


# =========================================================================
# 4. Multi-Layer Anomaly Detection
# =========================================================================

class MultiLayerAnomalyDetector:
    """Five-layer statistical anomaly detection for CEO-grade signal quality.
    
    Layers:
      1. Percentage Change — intuitive, fast
      2. Z-Score — parametric, assumes normal baseline
      3. IQR (Tukey Fences) — non-parametric, robust to outliers
      4. CUSUM — sequential drift detection (no false alarms on one-offs)
      5. Bayesian — P(real change | data), interpretable probability
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.previous_cusum_scores: Dict[str, float] = {}  # Persist across calls

    def detect(self, metric_facts: pd.DataFrame) -> pd.DataFrame:
        """Run multi-layer anomaly detection on weekly metric facts.

        Vectorized implementation: aggregates to weekly series per (table, metric),
        then computes rolling baseline statistics in a single groupby pass instead
        of looping with iterrows(). The CUSUM layer remains stateful per key (it
        accumulates across weeks for drift detection), but every other layer
        (pct_diff, z_score, IQR, Bayesian) is computed in O(n) over the weekly
        series.
        """
        logger.info("Running multi-layer anomaly detection...")

        METRIC_COLS = ["FUNDRAISERS", "GDV", "NAC2", "COMPLETES", "VIEWS", "CLICKS", "COST"]
        existing_metrics = [c for c in METRIC_COLS if c in metric_facts.columns]

        # Normalize to long format
        facts_long = metric_facts.melt(
            id_vars=["SOURCE_TABLE", "WEEK_START_DATE", "COUNTRY", "DEVICE", "TRAFFIC_SOURCE"],
            value_vars=existing_metrics,
            var_name="METRIC", value_name="VALUE"
        )
        facts_long["VALUE"] = pd.to_numeric(facts_long["VALUE"], errors="coerce").fillna(0)

        # Aggregate to one weekly series per (table, metric)
        weekly = (
            facts_long.groupby(["SOURCE_TABLE", "METRIC", "WEEK_START_DATE"], as_index=False)["VALUE"]
            .sum()
            .sort_values(["SOURCE_TABLE", "METRIC", "WEEK_START_DATE"])
            .reset_index(drop=True)
        )

        if weekly.empty:
            return pd.DataFrame()

        # ---- Vectorized rolling baseline (per group) ----
        g = weekly.groupby(["SOURCE_TABLE", "METRIC"], group_keys=False)
        baseline_window = self.config.baseline_weeks

        # Shift(1) drops the current row so the window is the N PRIOR weeks
        weekly["baseline_count"] = g["VALUE"].transform(
            lambda s: s.shift(1).rolling(window=baseline_window, min_periods=self.config.min_baseline_obs).count()
        )
        weekly["baseline_avg"] = g["VALUE"].transform(
            lambda s: s.shift(1).rolling(window=baseline_window, min_periods=self.config.min_baseline_obs).mean()
        )
        weekly["baseline_std"] = g["VALUE"].transform(
            lambda s: s.shift(1).rolling(window=baseline_window, min_periods=self.config.min_baseline_obs).std(ddof=1)
        )

        # ---- Vectorized layer 1-3, 5 stats ----
        safe_avg = weekly["baseline_avg"].replace(0, np.nan)
        safe_std = weekly["baseline_std"].replace(0, np.nan)

        weekly["pct_diff"] = ((weekly["VALUE"] - weekly["baseline_avg"]) / safe_avg) * 100
        weekly["z_score"] = (weekly["VALUE"] - weekly["baseline_avg"]) / safe_std

        # IQR via rolling quantile (per group)
        baseline_series = g["VALUE"].transform(
            lambda s: s.shift(1).rolling(window=baseline_window, min_periods=self.config.min_baseline_obs)
        )
        weekly["_q1"] = baseline_series.transform(lambda s: s.quantile(0.25))
        weekly["_q3"] = baseline_series.transform(lambda s: s.quantile(0.75))
        weekly["iqr"] = weekly["_q3"] - weekly["_q1"]
        iqr_lo = weekly["_q1"] - self.config.iqr_multiplier * weekly["iqr"]
        iqr_hi = weekly["_q3"] + self.config.iqr_multiplier * weekly["iqr"]
        weekly["iqr_outlier"] = (weekly["VALUE"] < iqr_lo) | (weekly["VALUE"] > iqr_hi)
        weekly = weekly.drop(columns=["_q1", "_q3", "iqr"])

        # Bayesian probability (vectorized; same formula as scalar version)
        se = weekly["baseline_std"] / np.sqrt(weekly["baseline_count"].clip(lower=1))
        t_stat = (weekly["VALUE"] - weekly["baseline_avg"]) / se.replace(0, np.nan)
        bf = np.exp(-0.5 * t_stat ** 2 / (1 + weekly["baseline_count"] * se ** 2 / (weekly["baseline_std"] ** 2).replace(0, np.nan)))
        weekly["bayes_change_prob"] = (1 / (1 + bf)).clip(0, 1).fillna(0.5)

        # ---- CUSUM (per-key state, vectorized within key) ----
        cusum_h = np.zeros(len(weekly))
        cusum_l = np.zeros(len(weekly))
        drift = self.config.cusum_drift
        threshold = self.config.cusum_threshold

        for key, idx in weekly.groupby(["SOURCE_TABLE", "METRIC"]).indices.items():
            prev_h, prev_l = self.previous_cusum_scores.get(key, (0.0, 0.0))
            vals = weekly["VALUE"].values[idx]
            avgs = weekly["baseline_avg"].values[idx]
            stds = weekly["baseline_std"].values[idx]
            cnts = weekly["baseline_count"].values[idx]
            n = len(idx)
            for i in range(n):
                if not np.isfinite(cnts[i]) or cnts[i] < self.config.min_baseline_obs or stds[i] == 0 or not np.isfinite(stds[i]):
                    cusum_h[idx[i]] = prev_h
                    cusum_l[idx[i]] = prev_l
                    continue
                delta = (vals[i] - avgs[i]) / stds[i]
                prev_h = max(0.0, prev_h + delta - drift)
                prev_l = max(0.0, prev_l - delta - drift)
                cusum_h[idx[i]] = prev_h
                cusum_l[idx[i]] = prev_l
            self.previous_cusum_scores[key] = (prev_h, prev_l)

        weekly["cusum_high"] = cusum_h
        weekly["cusum_low"] = cusum_l

        # ---- Layer votes (fully vectorized) ----
        pct = weekly["pct_diff"]
        z = weekly["z_score"]

        # Layer 1: pct_diff
        v1 = np.where(pct.notna(),
                      np.where(pct <= self.config.pct_drop_threshold, -1,
                               np.where(pct >= self.config.pct_spike_threshold, 1, 0)),
                      0)
        # Layer 2: z_score
        v2 = np.where(z.notna(),
                      np.where(z <= -self.config.z_score_threshold, -1,
                               np.where(z >= self.config.z_score_threshold, 1, 0)),
                      0)
        # Layer 3: IQR (sign = -1 if outlier and below mean, +1 if above)
        v3 = np.where(weekly["iqr_outlier"],
                      np.where(weekly["VALUE"] < weekly["baseline_avg"], -1, 1),
                      0)
        # Layer 4: CUSUM
        v4 = np.where(weekly["cusum_high"] > threshold, 1,
                      np.where(weekly["cusum_low"] > threshold, -1, 0))
        # Layer 5: Bayesian
        v5 = np.where(weekly["bayes_change_prob"] > self.config.bayesian_posterior_threshold,
                      np.where(weekly["VALUE"] < weekly["baseline_avg"], -1, 1),
                      0)

        weights = np.array([1.0, 2.0, 1.5, 1.0, 1.5])
        vote_matrix = np.column_stack([v1, v2, v3, v4, v5])
        weekly["ensemble_score"] = (vote_matrix * weights).sum(axis=1) / weights.sum()
        weekly["n_agreeing_layers"] = (vote_matrix != 0).sum(axis=1)
        weekly["consensus_strength"] = weekly["n_agreeing_layers"] / len(weights)

        # ---- Direction + severity (vectorized via numpy) ----
        weekly["direction"] = np.where(weekly["ensemble_score"] <= -0.3, "DROP",
                               np.where(weekly["ensemble_score"] >= 0.3, "SPIKE", "NORMAL"))

        weekly["severity"] = self._vectorized_severity(weekly)

        # ---- Drop helper columns, order output ----
        # weekly still has the UPPERCASE id columns from the melt (SOURCE_TABLE,
        # WEEK_START_DATE, METRIC) and lowercase computed columns. Normalize
        # the id columns to lowercase so downstream consumers (which read
        # .source_table, .week_start_date, .metric) see consistent casing.
        result_df = weekly.rename(columns={"VALUE": "actual"})
        result_df = result_df.rename(columns={
            "SOURCE_TABLE": "source_table",
            "WEEK_START_DATE": "week_start_date",
            "METRIC": "metric",
        }).drop(columns=[])
        result_df = result_df[[
            "source_table", "week_start_date", "metric", "actual",
            "baseline_avg", "baseline_std", "baseline_count",
            "pct_diff", "z_score", "iqr_outlier",
            "cusum_high", "cusum_low", "bayes_change_prob",
            "ensemble_score", "consensus_strength", "n_agreeing_layers",
            "direction", "severity",
        ]]
        logger.info(f"Detected anomalies across {result_df['source_table'].nunique()} tables, "
                     f"{len(result_df[result_df['direction'] != 'NORMAL'])} non-normal signals")
        return result_df

    def _vectorized_severity(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized replacement for _classify_severity (preserves the same rules)."""
        sev = np.full(len(df), "LOW", dtype=object)
        normal_mask = df["direction"] == "NORMAL"
        sev[normal_mask] = "LOW"

        non_normal = ~normal_mask
        pct_abs = df["pct_diff"].abs()
        z_abs = df["z_score"].abs()
        consensus = df["consensus_strength"]
        direction = df["direction"]

        critical_pct = pct_abs >= abs(self.config.pct_critical_drop)
        critical_z = z_abs >= self.config.z_score_critical
        high_consensus = consensus >= 0.6
        med_consensus = consensus >= 0.4

        is_critical = non_normal & (critical_pct.fillna(False) | critical_z.fillna(False))
        is_high = non_normal & ~is_critical & high_consensus & (direction == "DROP")
        is_med = non_normal & ~is_critical & ~is_high & (
            (high_consensus & (direction == "SPIKE")) | med_consensus
        )
        sev[is_critical] = AlertSeverity.CRITICAL.value
        sev[is_high] = AlertSeverity.HIGH.value
        sev[is_med] = AlertSeverity.MEDIUM.value
        return pd.Series(sev, index=df.index)

    def _cusum_score(self, baseline: np.ndarray, actual: float, key: str) -> Tuple[float, float]:
        """CUSUM: detect persistent drift from baseline mean.
        
        Returns (cusum_high, cusum_low) — cumulative sums above/below target.
        """
        target = baseline.mean()
        std = baseline.std(ddof=1) or 1.0
        
        # Initialize or retrieve previous CUSUM
        prev_high, prev_low = self.previous_cusum_scores.get(key, (0.0, 0.0))
        
        # Standardized deviation from target
        delta = (actual - target) / std
        
        # CUSUM update
        cusum_high = max(0, prev_high + delta - self.config.cusum_drift)
        cusum_low = max(0, prev_low - delta - self.config.cusum_drift)
        
        self.previous_cusum_scores[key] = (cusum_high, cusum_low)
        return cusum_high, cusum_low

    def _bayesian_change_probability(self, baseline: np.ndarray, actual: float) -> float:
        """Estimate P(real change | data) using a simple Bayesian model.
        
        Assumes baseline ~ N(μ, σ²), actual ~ N(μ + δ, σ²).
        Computes Bayes factor for δ ≠ 0 vs δ = 0 under conjugate prior.
        """
        if len(baseline) < 3:
            return 0.5  # Non-informative with too few data points
        
        mu = baseline.mean()
        sigma = baseline.std(ddof=1) or 1.0
        n = len(baseline)
        
        # Standard error of the mean
        se = sigma / np.sqrt(n)
        
        # Observe: (actual - mu) / se should be ~t(n-1) under H0 (no change)
        # Under H1 (real change), we use a Cauchy(0, r) prior on effect size
        # Approximate Bayes factor using BIC approximation
        t_stat = (actual - mu) / se
        bf = np.exp(-0.5 * t_stat**2 / (1 + n * se**2 / sigma**2))
        
        # Convert to posterior probability (assuming 50% prior on change)
        posterior = 1 / (1 + bf)
        return min(max(posterior, 0.0), 1.0)

    def _classify_severity(self, pct_diff, z_score, ensemble, consensus, direction) -> AlertSeverity:
        """Map anomaly metrics to CEO-facing severity level."""
        if direction == AnomalyDirection.NORMAL:
            return AlertSeverity.LOW
        if direction == AnomalyDirection.UNKNOWN:
            return AlertSeverity.INFO
        
        # Critical: large magnitude (in either direction) + strong consensus
        if (pd.notna(pct_diff) and abs(pct_diff) >= abs(self.config.pct_critical_drop)):
            return AlertSeverity.CRITICAL
        if (pd.notna(z_score) and abs(z_score) >= self.config.z_score_critical):
            return AlertSeverity.CRITICAL
        
        # High: significant movement + good consensus
        if consensus >= 0.6:
            if direction == AnomalyDirection.DROP:
                return AlertSeverity.HIGH
            return AlertSeverity.MEDIUM  # Spikes are less concerning than drops
        
        # Medium: notable but low consensus
        if consensus >= 0.4:
            return AlertSeverity.MEDIUM
        
        return AlertSeverity.LOW


# =========================================================================
# 5. Causal Decomposition Engine
# =========================================================================

class CausalDecompositionEngine:
    """Decomposes metric movements into causal factors.
    
    Decompositions:
      1. Volume vs Yield vs Mix — for GDV changes
      2. Funnel Stage Attribution — which stage is leaking
      3. Segment Contribution — which segment drove the change
      4. Leading Indicator Precedence — did a leading indicator predict this?
    """

    def __init__(self, config: DetectorConfig):
        self.config = config

    @staticmethod
    def _wide_to_long(metric_facts: pd.DataFrame) -> pd.DataFrame:
        """Convert wide weekly metrics (FUNDRAISERS/GDV/...) to long format with METRIC column.

        `WeeklyDataIngestor.ingest_all()` returns wide format. Several downstream
        engines (causal, financial) need long format with a METRIC column. This
        helper centralizes that transformation and ensures consistent column names
        even when a metric column is missing or fully null.
        """
        id_cols = ["SOURCE_TABLE", "WEEK_START_DATE", "COUNTRY", "DEVICE", "TRAFFIC_SOURCE"]
        metric_cols = ["FUNDRAISERS", "GDV", "NAC2", "COMPLETES", "VIEWS", "CLICKS", "COST"]
        existing = [c for c in metric_cols if c in metric_facts.columns]
        if not existing:
            return pd.DataFrame(columns=id_cols + ["METRIC", "VALUE"])
        long_df = metric_facts.melt(
            id_vars=id_cols,
            value_vars=existing,
            var_name="METRIC",
            value_name="VALUE",
        )
        long_df["VALUE"] = pd.to_numeric(long_df["VALUE"], errors="coerce").fillna(0)
        return long_df

    def decompose_gdv(self, metric_summary: pd.DataFrame,
                      metric_facts: pd.DataFrame) -> pd.DataFrame:
        """Decompose GDV changes into volume × price (yield) × mix effects.

        GDV = Donation Count × Avg Donation Value
        WoW Δ(GDV) ≈ Δ(Count) × Avg + Count × Δ(Avg) + Δ(Count) × Δ(Avg)

        Vectorized: aggregates current and prior windows per (source_table, week)
        in a single groupby pass, then computes all decomposition terms at once.
        """
        logger.info("Running GDV volume-yield-mix decomposition...")

        if metric_summary.empty or metric_facts.empty:
            return pd.DataFrame()

        latest_week = metric_summary["week_start_date"].max()
        if pd.isna(latest_week):
            return pd.DataFrame()

        # metric_facts arrives in WIDE format from WeeklyDataIngestor; convert to
        # long format so we can filter by METRIC == "GDV" / "COMPLETES".
        facts_long = self._wide_to_long(metric_facts)

        gdv_data = facts_long[facts_long["METRIC"] == "GDV"]
        complete_data = facts_long[facts_long["METRIC"] == "COMPLETES"]

        # Merge GDV + completes per week per table
        merged = gdv_data.merge(
            complete_data,
            on=["SOURCE_TABLE", "WEEK_START_DATE", "COUNTRY", "DEVICE", "TRAFFIC_SOURCE"],
            suffixes=("_gdv", "_completes"),
            how="inner"
        )

        if merged.empty:
            return pd.DataFrame()

        # Aggregate to per (table, week) totals — one pass.
        weekly = (
            merged.groupby(["SOURCE_TABLE", "WEEK_START_DATE"], as_index=False)
            .agg(GDV=("VALUE_gdv", "sum"), COMP=("VALUE_completes", "sum"))
        )
        weekly["avg_gift"] = np.where(weekly["COMP"] > 0, weekly["GDV"] / weekly["COMP"], 0.0)

        # Prior window per (table, week) using merge_asof-style logic via a
        # cross-join-then-filter, or simpler: for each (table, week) compute
        # the average GDV/COMP across the N weeks strictly before it.
        # Vectorize by sorting and computing rolling means SHIFTED by 1.
        weekly = weekly.sort_values(["SOURCE_TABLE", "WEEK_START_DATE"]).reset_index(drop=True)
        g = weekly.groupby("SOURCE_TABLE", group_keys=False)
        baseline_window = self.config.baseline_weeks

        # Shifted rolling mean (shift(1) drops the current row so the window is PRIOR weeks).
        weekly["prior_gdv"] = g["GDV"].transform(
            lambda s: s.shift(1).rolling(window=baseline_window, min_periods=1).mean()
        )
        weekly["prior_comp"] = g["COMP"].transform(
            lambda s: s.shift(1).rolling(window=baseline_window, min_periods=1).mean()
        )
        weekly["prior_avg_gift"] = np.where(
            weekly["prior_comp"] > 0, weekly["prior_gdv"] / weekly["prior_comp"], 0.0
        )

        # Decomposition formulas applied to whole vectors.
        comp_delta = weekly["COMP"] - weekly["prior_comp"]
        gift_delta = weekly["avg_gift"] - weekly["prior_avg_gift"]
        weekly["gdv_current"] = weekly["GDV"]
        weekly["gdv_baseline_avg"] = weekly["prior_gdv"]
        weekly["gdv_delta"] = weekly["GDV"] - weekly["prior_gdv"]
        weekly["completes_current"] = weekly["COMP"]
        weekly["completes_baseline_avg"] = weekly["prior_comp"]
        weekly["avg_gift_current"] = weekly["avg_gift"]
        weekly["avg_gift_baseline_avg"] = weekly["prior_avg_gift"]
        weekly["volume_effect"] = comp_delta * weekly["prior_avg_gift"]
        weekly["yield_effect"] = weekly["COMP"] * gift_delta
        weekly["interaction_effect"] = comp_delta * gift_delta

        nonzero = weekly["gdv_delta"] != 0
        weekly["volume_pct_of_change"] = np.where(
            nonzero, weekly["volume_effect"] / weekly["gdv_delta"] * 100, 0.0
        )
        weekly["yield_pct_of_change"] = np.where(
            nonzero, weekly["yield_effect"] / weekly["gdv_delta"] * 100, 0.0
        )

        # Narrative per row (cheap; num rows << original loop count).
        weekly["decomposition_narrative"] = [
            self._build_decomposition_narrative(
                v, y, d, ag, agp
            )
            for v, y, d, ag, agp in zip(
                weekly["volume_effect"], weekly["yield_effect"], weekly["gdv_delta"],
                weekly["avg_gift"], weekly["prior_avg_gift"],
            )
        ]

        return weekly[[
            "SOURCE_TABLE" if "SOURCE_TABLE" in weekly.columns else "source_table",
            "week_start_date", "gdv_current", "gdv_baseline_avg", "gdv_delta",
            "completes_current", "completes_baseline_avg",
            "avg_gift_current", "avg_gift_baseline_avg",
            "volume_effect", "yield_effect", "interaction_effect",
            "volume_pct_of_change", "yield_pct_of_change",
            "decomposition_narrative",
        ]].rename(columns={weekly.columns[0]: "source_table"})

    def _build_decomposition_narrative(self, volume_effect, yield_effect, 
                                        total_delta, avg_gift, avg_gift_prior) -> str:
        """Human-readable explanation of the decomposition."""
        abs_vol = abs(volume_effect)
        abs_yield = abs(yield_effect)
        
        if abs(total_delta) < self.config.weekly_impact_floor:
            return "Change below materiality threshold — no meaningful decomposition."
        
        if abs_vol > abs_yield:
            dominant = "volume (donation count)"
            secondary = "yield (average gift size)"
        else:
            dominant = "yield (average gift size)"
            secondary = "volume (donation count)"
        
        direction = "increase" if total_delta > 0 else "decline"
        
        return (
            f"GDV {direction} of ${abs(total_delta):,.0f} driven primarily by {dominant} "
            f"(${abs_vol:,.0f} vs ${abs_yield:,.0f} from {secondary}). "
            f"Average gift changed from ${avg_gift_prior:.2f} to ${avg_gift:.2f}."
        )

    def decompose_segment_contributions(self, metric_summary: pd.DataFrame,
                                        metric_facts: pd.DataFrame) -> pd.DataFrame:
        """Which segments (country/device/source) contributed most to metric changes?

        Vectorized: builds a single long table filtered to the latest week, then
        computes each segment's share of total with a groupby transform (one
        pass over the data instead of one per anomaly).
        """
        logger.info("Running segment contribution decomposition...")

        if metric_summary.empty or metric_facts.empty:
            return pd.DataFrame()

        latest_week = metric_summary["week_start_date"].max()
        if pd.isna(latest_week):
            return pd.DataFrame()

        latest = metric_summary[metric_summary["week_start_date"] == latest_week]
        if latest.empty:
            return pd.DataFrame()

        # metric_facts is WIDE; melt to long so we can filter by METRIC.
        facts_long = self._wide_to_long(metric_facts)

        # Build the keys that appear in `latest` for an isin-join, then keep only
        # the rows of facts_long that match any of those (table, metric) pairs AT
        # the latest week. A single mask + groupby replaces the nested iterrows.
        keys = list(zip(latest["source_table"], latest["metric"]))
        key_set = set(keys)
        # Vectorize membership check via merge on (source_table, metric).
        keys_df = pd.DataFrame(keys, columns=["source_table", "metric"])
        # Already have the melt; merge keeps the original column order.
        segment_data = facts_long[facts_long["WEEK_START_DATE"] == latest_week].merge(
            keys_df, on=["SOURCE_TABLE", "METRIC"],
            left_on=["SOURCE_TABLE", "METRIC"], right_on=["source_table", "metric"]
        )
        if segment_data.empty:
            return pd.DataFrame()

        # total VALUE per (source_table, metric) for the latest week
        totals = segment_data.groupby(["SOURCE_TABLE", "METRIC"])["VALUE"].transform("sum")
        segment_data = segment_data.assign(
            share_of_total_pct=np.where(totals > 0, segment_data["VALUE"] / totals * 100, 0.0)
        )
        segment_data = segment_data.rename(columns={
            "SOURCE_TABLE": "source_table",
            "METRIC": "metric",
            "WEEK_START_DATE": "week_start_date",
            "COUNTRY": "country",
            "DEVICE": "device",
            "TRAFFIC_SOURCE": "traffic_source",
        })
        # Drop the duplicate key cols from the right side of the merge.
        segment_data = segment_data.drop(columns=[c for c in segment_data.columns
                                                  if c.endswith("_x") or c.endswith("_y")])
        # The merge produces two 'source_table' / 'metric' columns from each side.
        # Keep the LEFT (facts_long) versions.
        out = segment_data[[
            "source_table", "week_start_date", "metric",
            "country", "device", "traffic_source", "VALUE", "share_of_total_pct",
        ]].rename(columns={"VALUE": "value"})
        return out.reset_index(drop=True)


# =========================================================================
# 6. Leading Indicator Engine
# =========================================================================

class LeadingIndicatorEngine:
    """Cross-correlation analysis to discover and score leading indicators.
    
    Identifies metrics that consistently precede changes in GDV or other
    target metrics. Example: NAC2 drops predict GDV drops 2 weeks later.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config

    def analyze(self, metric_summary: pd.DataFrame) -> pd.DataFrame:
        """Run cross-correlation between all metric pairs at specified lags (vectorized)."""
        logger.info("Running leading indicator analysis...")

        if metric_summary.empty:
            return pd.DataFrame()

        # Pivot to wide format: rows=weeks, cols=source_table::metric
        wide = metric_summary.pivot_table(
            index="week_start_date",
            columns=["source_table", "metric"],
            values="actual",
            aggfunc="sum"
        ).fillna(0)

        if len(wide) < 6:
            logger.warning("Insufficient weeks for leading indicator analysis")
            return pd.DataFrame()

        wide = wide.sort_index()
        wide.columns = ["::".join(c).upper() for c in wide.columns]

        gdv_cols = [c for c in wide.columns if "GDV" in c.upper()]
        other_cols = [c for c in wide.columns if "GDV" not in c.upper()]

        if not gdv_cols or not other_cols:
            logger.warning("Need both GDV and non-GDV metrics for leading indicator analysis")
            return pd.DataFrame()

        # Pre-compute predictor arrays shifted by each lag (one frame per lag).
        lags = [lag for lag in self.config.leading_indicator_lags if len(wide) > lag]
        if not lags:
            return pd.DataFrame()
        shifted_by_lag = {lag: wide[other_cols].shift(lag) for lag in lags}

        # Build a long table of (predictor, target, lag) combos and compute
        # correlation in one pass with np.corrcoef on aligned columns.
        records = []
        n = len(wide)
        for target in gdv_cols:
            target_arr = wide[target].to_numpy()
            for predictor in other_cols:
                pred_arr_base = wide[predictor].to_numpy()
                for lag in lags:
                    shifted = shifted_by_lag[lag][predictor].to_numpy()
                    # Use a mask of non-NaN pairs (numpy corrcoef doesn't accept NaN).
                    mask = ~(np.isnan(shifted) | np.isnan(target_arr))
                    n_valid = int(mask.sum())
                    if n_valid < 5:
                        continue
                    if not mask.all():
                        shifted = shifted[mask]
                        target_v = target_arr[mask]
                    else:
                        target_v = target_arr
                    # Guard against zero-variance inputs.
                    if shifted.std() == 0 or target_v.std() == 0:
                        corr = 0.0
                    else:
                        corr = float(np.corrcoef(shifted, target_v)[0, 1])

                    t_stat = corr * np.sqrt((n_valid - 2) / (1 - corr ** 2)) if abs(corr) < 1 else 0
                    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n_valid - 2))

                    records.append({
                        "predictor_metric": predictor,
                        "target_metric": target,
                        "lag_weeks": lag,
                        "correlation": corr,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "direction": "leading" if corr > 0 else "inverse_leading",
                        "n_observations": n_valid,
                    })

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            result_df = result_df.sort_values("correlation", ascending=False)
            sig = result_df[result_df["significant"]]
            logger.info(f"Found {len(sig)} significant leading indicator relationships")
        return result_df

    def generate_early_warnings(self, metric_summary: pd.DataFrame,
                                 leading_indicators: pd.DataFrame) -> pd.DataFrame:
        """Generate early warnings based on detected leading indicators (vectorized)."""
        if leading_indicators.empty or metric_summary.empty:
            return pd.DataFrame()

        significant = leading_indicators[
            (leading_indicators["significant"]) &
            (leading_indicators["direction"] == "leading")
        ].copy()
        if significant.empty:
            return pd.DataFrame()

        latest_week = metric_summary["week_start_date"].max()
        if pd.isna(latest_week):
            return pd.DataFrame()

        latest = metric_summary[metric_summary["week_start_date"] == latest_week]
        if latest.empty:
            return pd.DataFrame()

        # Parse the "TABLE::METRIC" predictor/target keys into two columns.
        split_pred = significant["predictor_metric"].str.split("::", n=1, expand=True)
        split_tgt = significant["target_metric"].str.split("::", n=1, expand=True)
        significant = significant.assign(
            pred_table=split_pred[0], pred_metric=split_pred[1],
            target_table=split_tgt[0], target_metric_name=split_tgt[1],
        ).dropna(subset=["pred_table", "pred_metric", "target_table", "target_metric_name"])

        # Join to the latest-week anomalies for the predictor.
        latest_lookup = latest[["source_table", "metric", "pct_diff", "direction", "baseline_avg"]]
        joined = significant.merge(
            latest_lookup,
            left_on=["pred_table", "pred_metric"],
            right_on=["source_table", "metric"],
            how="inner",
        )
        # Keep only anomalies with a current DROP/SPIKE.
        joined = joined[joined["direction"].isin(["DROP", "SPIKE"])]
        if joined.empty:
            return pd.DataFrame()

        # Projected impact.
        joined["projected_impact_pct"] = joined["pct_diff"] * joined["correlation"].abs() * 0.5
        joined["projected_impact_value"] = np.where(
            joined["baseline_avg"] != 0,
            joined["baseline_avg"] * joined["projected_impact_pct"] / 100,
            0.0,
        )
        joined = joined.rename(columns={"direction": "predictor_direction", "pct_diff": "predictor_pct_diff"})

        # Build narrative + recommended_action via vectorized f-strings over the rows.
        narr = []
        action = []
        for r in joined.to_dict("records"):
            narr.append(
                f"⚠ EARLY WARNING: {r['pred_table']}.{r['pred_metric']} "
                f"({r['predictor_direction']}: {r['predictor_pct_diff']:.1f}%) is a leading "
                f"indicator for {r['target_table']}.{r['target_metric_name']} "
                f"(lag={int(r['lag_weeks'])}wk, r={r['correlation']:.2f}). "
                f"If pattern holds, expect {r['projected_impact_pct']:.1f}% impact "
                f"(≈${abs(r['projected_impact_value']):,.0f}) in ~{int(r['lag_weeks'])} week(s)."
            )
            action.append(
                f"Investigate {r['pred_table']}.{r['pred_metric']} root cause now to mitigate "
                f"potential downstream impact on {r['target_table']}.{r['target_metric_name']}."
            )
        joined["narrative"] = narr
        joined["recommended_action"] = action
        joined["warning_type"] = "LEADING_INDICATOR"

        return joined[[
            "warning_type", "predictor_metric", "target_metric", "lag_weeks",
            "correlation", "predictor_pct_diff", "predictor_direction",
            "projected_impact_pct", "projected_impact_value",
            "narrative", "recommended_action",
        ]].reset_index(drop=True)


# =========================================================================
# 7. Financial Impact Estimator
# =========================================================================

class FinancialImpactEstimator:
    """Quantifies the financial impact of detected anomalies."""

    def __init__(self, config: DetectorConfig):
        self.config = config

    @staticmethod
    def _wide_to_long(metric_facts: pd.DataFrame) -> pd.DataFrame:
        """Convert wide weekly metrics to long format with METRIC column.

        `WeeklyDataIngestor.ingest_all()` returns wide format; financial impact
        estimation needs to look up ratios (avg gift, avg GDV per fundraiser)
        by METRIC, so we melt once here and reuse the long DataFrame below.
        """
        id_cols = ["SOURCE_TABLE", "WEEK_START_DATE", "COUNTRY", "DEVICE", "TRAFFIC_SOURCE"]
        metric_cols = ["FUNDRAISERS", "GDV", "NAC2", "COMPLETES", "VIEWS", "CLICKS", "COST"]
        existing = [c for c in metric_cols if c in metric_facts.columns]
        if not existing:
            return pd.DataFrame(columns=id_cols + ["METRIC", "VALUE"])
        long_df = metric_facts.melt(
            id_vars=id_cols,
            value_vars=existing,
            var_name="METRIC",
            value_name="VALUE",
        )
        long_df["VALUE"] = pd.to_numeric(long_df["VALUE"], errors="coerce").fillna(0)
        return long_df

    def estimate(self, metric_summary: pd.DataFrame,
                 metric_facts: pd.DataFrame) -> pd.DataFrame:
        """Estimate $ impact of non-normal anomalies (vectorized)."""
        logger.info("Estimating financial impact of anomalies...")

        if metric_summary.empty:
            return pd.DataFrame()

        # Pre-filter to rows that contribute to the report (drops/normal-low excluded).
        candidates = metric_summary[
            (metric_summary["direction"] != "NORMAL") &
            (metric_summary["severity"] != "LOW") &
            metric_summary["pct_diff"].notna() &
            metric_summary["actual"].notna() &
            metric_summary["baseline_avg"].notna()
        ].copy()
        if candidates.empty:
            return pd.DataFrame()

        candidates["dollar_impact"] = candidates["actual"] - candidates["baseline_avg"]

        # Precompute proxy multipliers from facts_long ONCE, per (source_table).
        gift_lookup = pd.Series(dtype=float)
        fund_lookup = pd.Series(dtype=float)
        if not metric_facts.empty:
            facts_long = self._wide_to_long(metric_facts)
            if not facts_long.empty:
                # Pivot to (SOURCE_TABLE, WEEK_START_DATE) x METRIC for fast lookup.
                pivot = facts_long.pivot_table(
                    index=["SOURCE_TABLE", "WEEK_START_DATE"],
                    columns="METRIC",
                    values="VALUE",
                    aggfunc="sum",
                )
                # avg_gift per (table, week) = GDV / COMPLETES
                gdv = pivot["GDV"] if "GDV" in pivot.columns else None
                comp = pivot["COMPLETES"] if "COMPLETES" in pivot.columns else None
                fund = pivot["FUNDRAISERS"] if "FUNDRAISERS" in pivot.columns else None
                if gdv is not None and comp is not None:
                    gift_lookup = (gdv / comp.replace(0, np.nan)).groupby(level=0).mean()
                if gdv is not None and fund is not None:
                    fund_lookup = (gdv / fund.replace(0, np.nan)).groupby(level=0).mean()

        # Apply per-metric multipliers vectorized.
        m_arr = candidates["metric"].to_numpy()
        dollar = candidates["dollar_impact"].to_numpy(dtype=float)

        # Look up proxy ratios; default to 1.0 when not present.
        gift_ratio = candidates["source_table"].map(gift_lookup).fillna(0).to_numpy()
        fund_ratio = candidates["source_table"].map(fund_lookup).fillna(0).to_numpy()
        safe_gift = np.where(gift_ratio > 0, gift_ratio, 1.0)
        safe_fund = np.where(fund_ratio > 0, fund_ratio, 1.0)

        out_dollar = np.zeros(len(candidates), dtype=float)
        out_dollar[m_arr == "GDV"] = dollar[m_arr == "GDV"]
        out_dollar[m_arr == "COMPLETES"] = dollar[m_arr == "COMPLETES"] * safe_gift[m_arr == "COMPLETES"]
        out_dollar[m_arr == "FUNDRAISERS"] = dollar[m_arr == "FUNDRAISERS"] * safe_fund[m_arr == "FUNDRAISERS"]
        out_dollar[m_arr == "VIEWS"] = dollar[m_arr == "VIEWS"] * 0.01
        out_dollar[m_arr == "CLICKS"] = dollar[m_arr == "CLICKS"] * 0.05
        out_dollar[m_arr == "COST"] = dollar[m_arr == "COST"]

        candidates["dollar_impact"] = out_dollar
        candidates["direction"] = np.where(
            m_arr == "COST", "COST_" + candidates["direction"].astype(object), candidates["direction"]
        )

        # Materiality filter
        candidates = candidates[candidates["dollar_impact"].abs() >= self.config.weekly_impact_floor]
        if candidates.empty:
            return pd.DataFrame()

        candidates["annualized_dollar_impact"] = candidates["dollar_impact"] * 52
        candidates["annualized_pct_of_target"] = (
            candidates["annualized_dollar_impact"] / self.config.annual_target_gdv * 100
        )
        # Build narrative strings in one vectorized pass.
        is_gdv = candidates["metric"] == "GDV"
        candidates["narrative"] = np.where(
            is_gdv, "Revenue impact", "Estimated proxy impact"
        )
        # Use list-comp for the f-string formatters (vectorize via zip).
        candidates["narrative"] = [
            f"{n}: ${abs(d):,.0f}/week (${abs(a):,.0f}/year "
            f"annualized, {abs(a)/self.config.annual_target_gdv*100:.1f}% of target). "
            f"Movement: {p:.1f}% ({dirn.lower()})."
            for n, d, a, p, dirn in zip(
                candidates["narrative"],
                candidates["dollar_impact"],
                candidates["annualized_dollar_impact"],
                candidates["pct_diff"],
                candidates["direction"],
            )
        ]

        result_df = candidates.rename(columns={"pct_diff": "pct_diff"})[
            ["source_table", "week_start_date", "metric", "direction", "severity",
             "pct_diff", "dollar_impact", "annualized_dollar_impact",
             "annualized_pct_of_target", "narrative"]
        ].rename(columns={"dollar_impact": "weekly_dollar_impact"})

        if not result_df.empty:
            total_annual = result_df["annualized_dollar_impact"].sum()
            logger.info(f"Total estimated annualized financial impact: ${total_annual:,.0f}")
        return result_df.reset_index(drop=True)


# =========================================================================
# 8. Executive Summary Engine
# =========================================================================

class ExecutiveSummaryEngine:
    """Generates CEO-ready narratives and recommendations.
    
    Produces:
      1. Executive brief (1-page summary)
      2. Ranked action items with expected ROI
      3. Traffic-light dashboard snapshot
      4. Trend narratives per metric family
      5. Early warning alerts
    """

    def __init__(self, config: DetectorConfig):
        self.config = config

    def generate(self, metric_summary: pd.DataFrame, financial_impacts: pd.DataFrame,
                 leading_warnings: pd.DataFrame, causal_decomp: pd.DataFrame,
                 segment_contributions: pd.DataFrame) -> dict:
        """Generate complete executive output package."""
        logger.info("Generating executive summary...")
        
        # --- 1. Traffic Light Dashboard ---
        dashboard = self._build_traffic_light(metric_summary, financial_impacts)
        
        # --- 2. Executive Brief ---
        brief = self._build_executive_brief(
            metric_summary, financial_impacts, leading_warnings, causal_decomp
        )
        
        # --- 3. Ranked Action Items ---
        actions = self._build_action_items(
            metric_summary, financial_impacts, leading_warnings
        )
        
        # --- 4. Early Warning Summary ---
        early_warnings = self._format_early_warnings(leading_warnings)
        
        # --- 5. Segment Spotlight ---
        segment_spotlight = self._build_segment_spotlight(segment_contributions)
        
        return {
            "traffic_light_dashboard": dashboard,
            "executive_brief": brief,
            "ranked_action_items": actions,
            "early_warnings": early_warnings,
            "segment_spotlight": segment_spotlight,
            "generated_at": pd.Timestamp.now(),
            "lookback_weeks": self.config.lookback_weeks,
            "baseline_weeks": self.config.baseline_weeks,
        }

    def _build_traffic_light(self, metric_summary: pd.DataFrame,
                              financial_impacts: pd.DataFrame) -> dict:
        """Build a CEO-facing traffic light dashboard (vectorized)."""
        if metric_summary.empty:
            return {"status": "NO_DATA", "summary": "No anomaly data available."}

        latest_week = metric_summary["week_start_date"].max()
        latest = metric_summary[metric_summary["week_start_date"] == latest_week]

        critical_mask = latest["severity"] == "CRITICAL"
        high_mask = latest["severity"] == "HIGH"
        medium_mask = latest["severity"] == "MEDIUM"
        critical = latest[critical_mask]
        high = latest[high_mask]
        medium = latest[medium_mask]

        total_annual_impact = 0
        if not financial_impacts.empty:
            total_annual_impact = float(financial_impacts["annualized_dollar_impact"].sum())

        if critical_mask.any():
            overall = "🔴 CRITICAL"
        elif high_mask.any():
            overall = "🟡 WARNING"
        elif medium_mask.any():
            overall = "🟢 MONITOR"
        else:
            overall = "✅ HEALTHY"

        # Build dicts from the filtered frames in one .to_dict call each.
        critical_items = [
            {
                "metric": r["metric"],
                "table": r["source_table"],
                "pct_change": round(r["pct_diff"], 1),
                "severity": r["severity"],
                "z_score": round(r["z_score"], 2) if pd.notna(r.get("z_score")) else None,
            }
            for r in critical.to_dict("records")
        ]
        high_items = [
            {
                "metric": r["metric"],
                "table": r["source_table"],
                "pct_change": round(r["pct_diff"], 1),
                "severity": r["severity"],
            }
            for r in high.to_dict("records")
        ]

        return {
            "overall_status": overall,
            "reporting_week": str(latest_week.date()) if pd.notna(latest_week) else "Unknown",
            "critical_issues": int(critical_mask.sum()),
            "high_issues": int(high_mask.sum()),
            "medium_issues": int(medium_mask.sum()),
            "total_anomalies": int((latest["direction"] != "NORMAL").sum()),
            "annualized_financial_exposure": total_annual_impact,
            "critical_items": critical_items,
            "high_items": high_items,
        }

    def _build_executive_brief(self, metric_summary, financial_impacts,
                                leading_warnings, causal_decomp) -> str:
        """Generate a 1-page CEO brief narrative (vectorized)."""
        if metric_summary.empty:
            return "No data available for executive briefing."

        latest_week = metric_summary["week_start_date"].max()
        latest = metric_summary[metric_summary["week_start_date"] == latest_week]

        # Pre-filter the four sub-views once.
        gdv_anomalies = latest[latest["metric"] == "GDV"]
        critical = latest[latest["severity"] == "CRITICAL"]
        high = latest[latest["severity"] == "HIGH"]
        medium = latest[latest["severity"] == "MEDIUM"]

        lines = []
        lines.append("=" * 72)
        lines.append(f"  EXECUTIVE WEEKLY PERFORMANCE BRIEF")
        lines.append(f"  Week Ending: {str(latest_week.date()) if pd.notna(latest_week) else 'Unknown'}")
        lines.append("=" * 72)
        lines.append("")

        # --- Top-line summary ---
        if not gdv_anomalies.empty:
            top_row = gdv_anomalies.sort_values("pct_diff").iloc[0]
            lines.append(f"  TOP-LINE: GDV {top_row['direction']} of {top_row['pct_diff']:.1f}% "
                         f"(z={top_row['z_score']:.1f}, severity={top_row['severity']})")
            lines.append(f"  Source: {top_row['source_table']}")
        else:
            lines.append("  TOP-LINE: GDV within normal range.")
        lines.append("")

        # --- Causal decomposition (use to_dict('records')) ---
        if not causal_decomp.empty:
            latest_decomp = causal_decomp[causal_decomp["week_start_date"] == latest_week]
            for d in latest_decomp.to_dict("records"):
                lines.append(f"  DECOMP: {d['decomposition_narrative']}")
            lines.append("")

        # --- Critical issues ---
        if not critical.empty:
            lines.append("  ⛔ CRITICAL ISSUES (CEO attention required)")
            for r in critical.to_dict("records"):
                lines.append(f"    • {r['source_table']}.{r['metric']}: "
                             f"{r['pct_diff']:.1f}% ({r['direction']})")
            lines.append("")

        # --- High issues ---
        if not high.empty:
            lines.append("  ⚠  HIGH PRIORITY (Functional leader attention)")
            for r in high.to_dict("records"):
                lines.append(f"    • {r['source_table']}.{r['metric']}: "
                             f"{r['pct_diff']:.1f}% ({r['direction']})")
            lines.append("")

        # --- Financial impact ---
        if not financial_impacts.empty:
            total_annual = float(financial_impacts["annualized_dollar_impact"].sum())
            total_weekly = float(financial_impacts["weekly_dollar_impact"].sum())
            # Revenue-side impact: a positive weekly delta is favorable (more $),
            # a negative weekly delta is unfavorable (less $). For COST-tagged
            # rows the estimator already flipped the sign so the same convention
            # applies (cost down = positive dollar_impact = favorable).
            lines.append(f"  💰 FINANCIAL IMPACT ESTIMATE")
            lines.append(f"    Weekly: ${abs(total_weekly):,.0f} "
                         f"({'favorable' if total_weekly > 0 else 'unfavorable'})")
            lines.append(f"    Annualized run-rate: ${abs(total_annual):,.0f} "
                         f"({'opportunity' if total_annual > 0 else 'exposure'})")
            lines.append(f"    % of Annual Target: {abs(total_annual)/self.config.annual_target_gdv*100:.1f}%")
            lines.append("")

        # --- Leading indicator warnings ---
        if not leading_warnings.empty:
            lines.append("  🔮 EARLY WARNINGS (Leading Indicators)")
            for w in leading_warnings.to_dict("records"):
                lines.append(f"    • {w['narrative']}")
            lines.append("")

        # --- Segment spotlight ---
        lines.append("  📊 KEY TAKEAWAYS")
        lines.append(f"    • {len(critical)} critical, {len(high)} high, "
                     f"{len(medium)} medium signals detected")
        lines.append(f"    • Data spans {metric_summary['week_start_date'].nunique()} weeks "
                     f"across {metric_summary['source_table'].nunique()} source tables")
        lines.append("")
        lines.append("=" * 72)

        return "\n".join(lines)

    # Metric → (action, category, owner) lookup table. Order matters: more
    # specific keys first (VIEWS/CLICKS share a branch, so they go first).
    _ACTION_LOOKUP = {
        "VIEWS": ("Investigate traffic sources, paid acquisition efficiency, and channel mix",
                  "Traffic & Acquisition", "Growth / Marketing"),
        "CLICKS": ("Investigate traffic sources, paid acquisition efficiency, and channel mix",
                   "Traffic & Acquisition", "Growth / Marketing"),
        "COMPLETES": ("Audit donation flow conversion, checkout UX, and payment failures",
                      "Donor Conversion", "Product / Engineering"),
        "GDV": ("Decompose into donation volume vs average gift; check segment contributions",
                "Revenue", "CFO / FP&A"),
        "NAC2": ("Review fundraiser quality, onboarding flow, and early donor activation",
                 "Fundraiser Quality", "Product / Growth"),
        "FUNDRAISERS": ("Analyze publish funnels by country, device, and traffic source",
                        "Supply", "Growth / Product"),
        "COST": ("Review paid marketing efficiency, CPC trends, and ROAS by channel",
                 "Marketing Efficiency", "Marketing"),
    }
    _DEFAULT_ACTION = ("Investigate root cause at source table level", "General", "Data / Analytics")

    def _build_action_items(self, metric_summary, financial_impacts,
                             leading_warnings) -> pd.DataFrame:
        """Rank actionable recommendations by expected ROI (vectorized)."""
        if metric_summary.empty:
            return pd.DataFrame()

        latest_week = metric_summary["week_start_date"].max()
        latest = metric_summary[metric_summary["week_start_date"] == latest_week]
        anomalous = latest[latest["direction"] != "NORMAL"].copy()

        if anomalous.empty:
            return pd.DataFrame([{
                "priority": "LOW",
                "metric": "—",
                "source_table": "—",
                "pct_change": 0.0,
                "direction": "NORMAL",
                "action": "No action required — all metrics normal.",
                "expected_impact": "$0",
                "financial_annualized": 0,
                "category": "General",
                "owner": "Monitor",
                "urgency": "None",
                "consensus_strength": 0.0,
            }])

        # Join financial impact in one merge, then map metric → action via lookup.
        if not financial_impacts.empty:
            fi = financial_impacts[["source_table", "metric", "annualized_dollar_impact"]].drop_duplicates(
                ["source_table", "metric"]
            )
            merged = anomalous.merge(fi, on=["source_table", "metric"], how="left")
        else:
            merged = anomalous.copy()
            merged["annualized_dollar_impact"] = 0.0
        merged["annualized_dollar_impact"] = merged["annualized_dollar_impact"].fillna(0.0)

        # Vectorized metric → (action, category, owner) via a small frame and merge.
        lookup_df = pd.DataFrame(
            [(k, *v) for k, v in self._ACTION_LOOKUP.items()],
            columns=["metric", "action", "category", "owner"],
        )
        default_df = pd.DataFrame(
            [("", *self._DEFAULT_ACTION)],
            columns=["metric", "action", "category", "owner"],
        )
        lookup_df = pd.concat([lookup_df, default_df], ignore_index=True)
        merged = merged.merge(lookup_df, on="metric", how="left", suffixes=("", "_lkp"))
        # If merge missed (metric not in lookup + not default), fall back.
        merged["action"] = merged["action"].fillna(self._DEFAULT_ACTION[0])
        merged["category"] = merged["category"].fillna(self._DEFAULT_ACTION[1])
        merged["owner"] = merged["owner"].fillna(self._DEFAULT_ACTION[2])
        # Strip any duplicate _lkp columns
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_lkp")])

        merged["expected_impact"] = np.where(
            merged["annualized_dollar_impact"] != 0,
            "$" + merged["annualized_dollar_impact"].abs().map(lambda v: f"{v:,.0f}") + "/yr",
            "Unquantified",
        )
        merged["urgency"] = np.where(
            merged["severity"].isin(["CRITICAL", "HIGH"]), "This week", "Normal cadence"
        )
        merged["pct_change"] = merged["pct_diff"].round(1)
        merged["consensus_strength"] = merged["consensus_strength"].round(2)

        result = merged.rename(columns={"annualized_dollar_impact": "financial_annualized"})[
            ["severity", "metric", "source_table", "pct_change", "direction",
             "action", "expected_impact", "financial_annualized",
             "category", "owner", "urgency", "consensus_strength"]
        ].rename(columns={"severity": "priority"})

        result = result.sort_values(
            by=["priority", "financial_annualized"],
            ascending=[True, False],  # CRITICAL first, then highest $
        ).reset_index(drop=True)
        return result

    def _format_early_warnings(self, leading_warnings: pd.DataFrame) -> list:
        """Format leading indicator warnings for executive consumption (vectorized)."""
        if leading_warnings.empty:
            return [{"message": "No leading indicator warnings detected.", "count": 0}]
        return [
            {
                "message": r["narrative"],
                "action": r["recommended_action"],
                "lag_weeks": int(r["lag_weeks"]),
                "projected_impact_value": float(r["projected_impact_value"]),
            }
            for r in leading_warnings.to_dict("records")
        ]

    def _build_segment_spotlight(self, segment_contributions: pd.DataFrame) -> str:
        """Highlight key segment-level insights (vectorized)."""
        if segment_contributions.empty:
            return "No segment-level data available for this analysis."

        top5 = (
            segment_contributions
            .sort_values("share_of_total_pct", ascending=False)
            .head(5)
            .to_dict("records")
        )
        lines = ["Segment Contribution Spotlight:"]
        for r in top5:
            lines.append(
                f"  • {r['metric']} / {r['source_table']}: "
                f"{r['country'] or 'Global'} | "
                f"{r['device'] or 'All devices'} | "
                f"{r['traffic_source'] or 'All sources'} = "
                f"{r['share_of_total_pct']:.1f}% of total"
            )
        return "\n".join(lines)


# =========================================================================
# 9. Seasonality & YoY Comparison
# =========================================================================

class SeasonalityAnalyzer:
    """Add year-over-year context to anomaly detection."""

    def __init__(self, config: DetectorConfig):
        self.config = config

    def analyze(self, metric_summary: pd.DataFrame) -> pd.DataFrame:
        """Compute YoY comparison for each metric."""
        if not self.config.enable_yoy_comparison or metric_summary.empty:
            return pd.DataFrame()
        
        logger.info("Running YoY seasonality analysis...")
        
        yoy = metric_summary.copy()
        yoy["yoy_week"] = yoy["week_start_date"] - pd.Timedelta(weeks=52)
        
        merged = yoy.merge(
            metric_summary[["source_table", "metric", "week_start_date", "actual", "baseline_avg"]],
            left_on=["source_table", "metric", "yoy_week"],
            right_on=["source_table", "metric", "week_start_date"],
            suffixes=("", "_yoy"),
            how="inner"
        )
        
        if merged.empty:
            return pd.DataFrame()
        
        merged["yoy_pct_diff"] = (
            (merged["actual"] - merged["actual_yoy"]) / merged["actual_yoy"].replace(0, np.nan) * 100
        )
        
        merged = merged.dropna(subset=["yoy_pct_diff"])
        
        result = merged[[
            "source_table", "metric", "week_start_date",
            "actual", "actual_yoy", "yoy_pct_diff"
        ]].copy()
        
        logger.info(f"YoY comparison available for {len(result)} data points")
        return result


# =========================================================================
# 10. Persistence & Lineage Tracking
# =========================================================================

class OutputManager:
    """Handles Snowflake persistence, data lineage, and audit trails."""

    def __init__(self, session, config: DetectorConfig):
        self.session = session
        self.config = config

    def save(self, df: pd.DataFrame, table_name: str, description: str = "") -> str:
        """Save a DataFrame to Snowflake with lineage metadata.
        
        Returns the fully qualified table name.
        """
        if df.empty:
            logger.warning(f"Skipping save for {table_name} — empty DataFrame")
            return f"{self.config.database}.{self.config.schema}.{table_name}"
        
        full_name = f"{self.config.database}.{self.config.schema}.{table_name}"
        
        logger.info(f"Saving {len(df)} rows → {full_name}")
        
        self.session.write_pandas(
            df,
            table_name=table_name,
            database=self.config.database,
            schema=self.config.schema,
            auto_create_table=True,
            overwrite=True
        )
        
        # Write lineage metadata to a comment on the table
        try:
            self.session.sql(f"""
                ALTER TABLE {full_name} SET COMMENT = '{json.dumps({
                    "description": description,
                    "generated_by": "EnterpriseAnomalyDetector v2.0",
                    "generated_at": str(pd.Timestamp.now()),
                    "lookback_weeks": self.config.lookback_weeks,
                    "baseline_weeks": self.config.baseline_weeks,
                    "row_count": len(df),
                    "columns": list(df.columns),
                }).replace("'", "''")}'
            """).collect()
        except Exception as e:
            logger.warning(f"Could not set table comment for {full_name}: {e}")
        
        return full_name


# =========================================================================
# 11. Main Orchestrator
# =========================================================================

class EnterpriseAnomalyDetector:
    """Primary orchestrator — coordinates all subsystems into a single pipeline.
    
    Usage:
        detector = EnterpriseAnomalyDetector(session)
        results = detector.run()
        results["executive_brief"]  # CEO narrative
        results["action_items"]     # Prioritized actions
        results["dashboard"]        # Traffic light
    """

    def __init__(self, session,
                 config: Optional[DetectorConfig] = None):
        self.session = session
        self.config = config or DetectorConfig()
        
        # Subsystems
        self.schema = SchemaDiscovery(session, self.config)
        self.quality = DataQualityGate(session, self.config, self.schema)
        self.ingestor = WeeklyDataIngestor(session, self.config, self.schema)
        self.detector = MultiLayerAnomalyDetector(self.config)
        self.causal = CausalDecompositionEngine(self.config)
        self.leading = LeadingIndicatorEngine(self.config)
        self.financial = FinancialImpactEstimator(self.config)
        self.executive = ExecutiveSummaryEngine(self.config)
        self.seasonality = SeasonalityAnalyzer(self.config)
        self.output = OutputManager(session, self.config)
        
        # State
        self._columns_df = None
        self._table_profile = None
        self._metric_facts = None
        self._metric_summary = None

    def run(self, save_outputs: bool = False) -> dict:
        """Execute the full enterprise analysis pipeline.
        
        Returns a dict with all outputs for consumption.
        """
        logger.info("=" * 60)
        logger.info(" EnterpriseAnomalyDetector — Starting Pipeline")
        logger.info("=" * 60)
        
        # --- Step 1: Schema Discovery ---
        logger.info("Step 1/8: Schema Discovery")
        self._columns_df = self.schema.discover()
        self._table_profile = self.schema.build_profile()
        
        # --- Step 2: Data Quality Gate ---
        logger.info("Step 2/8: Data Quality Validation")
        quality_pass = self.quality.validate()
        if not quality_pass:
            logger.error("Pipeline aborted: data quality check failed")
            return {"error": "Data quality gate failed — check logs for details"}
        
        # --- Step 3: Data Ingestion ---
        logger.info("Step 3/8: Weekly Data Ingestion")
        self._metric_facts = self.ingestor.ingest_all()
        
        # --- Step 4: Multi-layer Anomaly Detection ---
        logger.info("Step 4/8: Multi-Layer Anomaly Detection")
        self._metric_summary = self.detector.detect(self._metric_facts)
        
        # --- Step 5: Causal Decomposition ---
        logger.info("Step 5/8: Causal Decomposition")
        causal_decomp = self.causal.decompose_gdv(
            self._metric_summary, self._metric_facts
        )
        segment_contributions = self.causal.decompose_segment_contributions(
            self._metric_summary, self._metric_facts
        )
        
        # --- Step 6: Leading Indicator Analysis ---
        logger.info("Step 6/8: Leading Indicator Analysis")
        leading_indicators = self.leading.analyze(self._metric_summary)
        leading_warnings = self.leading.generate_early_warnings(
            self._metric_summary, leading_indicators
        )
        
        # --- Step 7: Financial Impact ---
        logger.info("Step 7/8: Financial Impact Estimation")
        financial_impacts = self.financial.estimate(
            self._metric_summary, self._metric_facts
        )
        
        # --- Step 8: Executive Summary ---
        logger.info("Step 8/8: Executive Summary Generation")
        executive = self.executive.generate(
            self._metric_summary, financial_impacts,
            leading_warnings, causal_decomp, segment_contributions
        )
        
        # --- Optional: Seasonality ---
        yoy = self.seasonality.analyze(self._metric_summary)
        
        # --- Display (no Snowflake writes) ---
        logger.info("Displaying outputs...")
        print("\n" + "="*72)
        print("  📊 ENTERPRISE ANOMALY DETECTOR — RESULTS")
        print("="*72)
        print(f"\n🚦 DASHBOARD: {executive['traffic_light_dashboard']['overall_status']}")
        print(f"   Week: {executive['traffic_light_dashboard']['reporting_week']}")
        print(f"   Critical: {executive['traffic_light_dashboard']['critical_issues']} | "
              f"High: {executive['traffic_light_dashboard']['high_issues']} | "
              f"Medium: {executive['traffic_light_dashboard']['medium_issues']}")
        
        print("\n📋 ACTION ITEMS (ranked by priority × $ impact):")
        actions_df = executive["ranked_action_items"]
        if not actions_df.empty:
            for item in actions_df.to_dict("records"):
                print(f"  [{item['priority']:8s}] {item['metric']:12s} | "
                      f"{item['action']} | ${item.get('financial_annualized',0):,.0f}/yr | "
                      f"Owner: {item['owner']} | Urgency: {item['urgency']}")

        print("\n🔮 EARLY WARNINGS:")
        for w in executive["early_warnings"]:
            print(f"  • {w.get('message', w.get('narrative', 'N/A'))}")
        
        print(f"\n💰 FINANCIAL IMPACT: "
              f"${abs(financial_impacts['annualized_dollar_impact'].sum()):,.0f}/yr annualized"
              if not financial_impacts.empty else "\n💰 FINANCIAL IMPACT: None above materiality threshold")
        
        print(f"\n📈 Data spans {self._metric_summary['week_start_date'].nunique() if not self._metric_summary.empty else 0} weeks "
              f"across {self._metric_summary['source_table'].nunique() if not self._metric_summary.empty else 0} source tables\n")
        
        logger.info("=" * 60)
        logger.info(" Pipeline Complete — Executive Summary Ready")
        logger.info("=" * 60)
        
        return {
            # Core data
            "metric_facts": self._metric_facts,
            "metric_summary": self._metric_summary,
            "causal_decomposition": causal_decomp,
            "segment_contributions": segment_contributions,
            "leading_indicators": leading_indicators,
            "leading_warnings": leading_warnings,
            "financial_impacts": financial_impacts,
            "yoy_comparison": yoy,
            
            # Executive output
            "dashboard": executive["traffic_light_dashboard"],
            "executive_brief": executive["executive_brief"],
            "action_items": executive["ranked_action_items"],
            "early_warnings": executive["early_warnings"],
            "segment_spotlight": executive["segment_spotlight"],
            
            # Metadata
            "generated_at": executive["generated_at"],
            "config": asdict(self.config),
        }

    def print_executive_brief(self):
        """Convenience method to print the executive brief to console."""
        results = self.run(save_outputs=False)
        brief = results.get("executive_brief", "No executive brief generated.")
        print(brief)
        return results


# =========================================================================
# 12. Quick-start entry point
# =========================================================================

if __name__ == "__main__":
    # Run with defaults — displays to console, no Snowflake writes
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    
    detector = EnterpriseAnomalyDetector(session)
    results = detector.run(save_outputs=False)
    
    # Executive brief is already printed by run()