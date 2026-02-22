#!/usr/bin/env python3
"""
Marketing Campaign MCP Server

Exposes marketing_campaign.csv as analytical tools via MCP stdio transport.
Claude Desktop launches this as a subprocess and communicates over stdin/stdout.
"""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─── Constants ────────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "marketing_campaign.csv"

SPEND_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]
PURCHASE_COLS = [
    "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumDealsPurchases",
]
CAMPAIGN_COLS = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
    "AcceptedCmp4", "AcceptedCmp5", "Response",
]
ANOMALOUS_MARITAL = {"Alone", "YOLO", "Absurd"}

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load the CSV once at startup. Uses utf-8-sig to strip the BOM on the ID column."""
    return pd.read_csv(
        CSV_PATH,
        sep=";",
        encoding="utf-8-sig",
        parse_dates=["Dt_Customer"],
    )


DF = load_data()  # module-level singleton; all handlers share this immutable frame

# ─── Helper Utilities ─────────────────────────────────────────────────────────

def normalize_marital(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Alone/YOLO/Absurd collapsed into Single."""
    df = df.copy()
    df["Marital_Status"] = df["Marital_Status"].replace(
        {k: "Single" for k in ANOMALOUS_MARITAL}
    )
    return df


def safe_float(v: Any) -> Any:
    """Replace NaN/Inf with None for JSON safety."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def records_from_df(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame (possibly with MultiIndex) to JSON-safe records."""
    raw = df.reset_index().to_dict(orient="records")
    cleaned = []
    for row in raw:
        cleaned.append({
            k: safe_float(float(v)) if isinstance(v, float) else
               int(v) if hasattr(v, "item") else v
            for k, v in row.items()
        })
    return cleaned


# ─── MCP Server ───────────────────────────────────────────────────────────────

app = Server("marketing-mcp")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_dataset_overview",
            description=(
                "Returns a high-level summary of the marketing dataset: total row count, "
                "column names with data types, missing value counts, numeric ranges, and "
                "unique values for categorical fields. Run this first to understand the data."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get_spending_by_segment",
            description=(
                "Returns average spending per product category (Wines, Fruits, Meat, Fish, "
                "Sweets, Gold) broken down by a demographic segment. Reveals which customer "
                "groups spend the most on which product types."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_by": {
                        "type": "string",
                        "enum": ["Education", "Marital_Status"],
                        "description": "Demographic dimension to group customers by.",
                    },
                    "normalize_marital_status": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "If true (default), collapses Alone/YOLO/Absurd into Single "
                            "for cleaner grouping."
                        ),
                    },
                },
                "required": ["segment_by"],
            },
        ),
        types.Tool(
            name="get_campaign_performance",
            description=(
                "Returns acceptance rates (%) for all 6 marketing campaigns (Cmp1–5 and "
                "the final Response campaign), with optional breakdown by a demographic "
                "segment. Response is the most recent campaign."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_by": {
                        "type": "string",
                        "enum": ["Education", "Marital_Status", "none"],
                        "default": "none",
                        "description": (
                            "Demographic dimension to break down campaign rates by. "
                            "Use 'none' for overall rates only."
                        ),
                    },
                    "normalize_marital_status": {
                        "type": "boolean",
                        "default": True,
                        "description": "Collapse Alone/YOLO/Absurd into Single.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_channel_analysis",
            description=(
                "Returns purchase channel distribution: what share of all purchases go "
                "through web, catalog, store, and deal channels. Also shows average "
                "monthly web visits. Optionally compares channels across segments."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "segment_by": {
                        "type": "string",
                        "enum": ["Education", "Marital_Status", "has_children", "none"],
                        "default": "none",
                        "description": (
                            "Group channel stats by segment. 'has_children' groups by "
                            "whether the household has any children or teenagers."
                        ),
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_rfm_segments",
            description=(
                "Segments customers using RFM (Recency, Frequency, Monetary) scoring "
                "into tiers. Returns the size, average characteristics, and campaign "
                "response rate of each RFM segment. Higher R=more recent, higher F=more "
                "purchases, higher M=more spend."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n_tiers": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 2,
                        "maximum": 5,
                        "description": (
                            "Number of tiers per dimension (3 = Low/Mid/High, "
                            "5 = quintiles)."
                        ),
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_income_spend_correlation",
            description=(
                "Analyzes the relationship between customer income and total spending. "
                "Returns the Pearson correlation coefficient and a table of income brackets "
                "with average spend per bracket. Excludes the 24 customers with missing income."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "income_brackets": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 3,
                        "maximum": 10,
                        "description": "Number of equal-width income brackets to create.",
                    },
                    "exclude_income_outliers": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "If true (default), excludes customers with income > 150,000 "
                            "(likely data anomalies) before bracketing."
                        ),
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_customer_tenure_analysis",
            description=(
                "Analyzes customer behavior relative to how long they have been registered. "
                "Groups customers into tenure cohorts and compares average spending, recency, "
                "and campaign response rate across cohorts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cohort_months": {
                        "type": "integer",
                        "default": 6,
                        "minimum": 1,
                        "maximum": 12,
                        "description": (
                            "Width of each tenure cohort in months "
                            "(e.g. 6 = 0–6mo, 6–12mo, 12–18mo, ...)."
                        ),
                    },
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    handlers: dict[str, Any] = {
        "get_dataset_overview": handle_dataset_overview,
        "get_spending_by_segment": handle_spending_by_segment,
        "get_campaign_performance": handle_campaign_performance,
        "get_channel_analysis": handle_channel_analysis,
        "get_rfm_segments": handle_rfm_segments,
        "get_income_spend_correlation": handle_income_spend_correlation,
        "get_customer_tenure_analysis": handle_customer_tenure_analysis,
    }
    if name not in handlers:
        raise ValueError(f"Unknown tool: {name}")

    result = await handlers[name](arguments or {})
    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ─── Tool Handlers ────────────────────────────────────────────────────────────

async def handle_dataset_overview(args: dict) -> dict:
    df = DF
    columns = []
    for col in df.columns:
        info: dict[str, Any] = {"name": col}
        if pd.api.types.is_numeric_dtype(df[col]):
            info["type"] = "numeric"
            info["missing"] = int(df[col].isna().sum())
            non_null = df[col].dropna()
            info["min"] = round(float(non_null.min()), 2)
            info["max"] = round(float(non_null.max()), 2)
            info["mean"] = round(float(non_null.mean()), 2)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            info["type"] = "date"
            info["missing"] = int(df[col].isna().sum())
            info["min"] = str(df[col].min().date())
            info["max"] = str(df[col].max().date())
        else:
            info["type"] = "categorical"
            info["missing"] = int(df[col].isna().sum())
            info["unique_count"] = int(df[col].nunique())
            if df[col].nunique() <= 12:
                info["unique_values"] = sorted(df[col].dropna().unique().tolist())
        columns.append(info)

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": columns,
        "data_quality_notes": [
            f"{int(df['Income'].isna().sum())} customers are missing Income "
            "(excluded from income-based analyses by default)",
            f"Anomalous Marital_Status values present: {sorted(ANOMALOUS_MARITAL)} "
            "— use normalize_marital_status=true in other tools to merge these into 'Single'",
        ],
    }


async def handle_spending_by_segment(args: dict) -> dict:
    segment_by = args["segment_by"]
    normalize = args.get("normalize_marital_status", True)

    df = normalize_marital(DF) if normalize else DF.copy()
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)

    agg = df.groupby(segment_by)[SPEND_COLS + ["TotalSpend"]].mean().round(2)
    counts = df.groupby(segment_by).size().rename("customer_count")
    merged = agg.join(counts)

    segments = []
    for seg_name, row in merged.iterrows():
        segments.append({
            "segment": seg_name,
            "customer_count": int(row["customer_count"]),
            "avg_spend": {
                col: round(float(row[col]), 2)
                for col in SPEND_COLS + ["TotalSpend"]
            },
        })
    segments.sort(key=lambda x: x["avg_spend"]["TotalSpend"], reverse=True)

    return {
        "segment_by": segment_by,
        "normalized_marital_status": normalize,
        "segments": segments,
    }


async def handle_campaign_performance(args: dict) -> dict:
    segment_by = args.get("segment_by", "none")
    normalize = args.get("normalize_marital_status", True)

    df = normalize_marital(DF) if normalize else DF.copy()
    df["AnyAccepted"] = df[CAMPAIGN_COLS].any(axis=1).astype(int)

    overall_rates = (df[CAMPAIGN_COLS].mean() * 100).round(2).to_dict()
    overall_rates["any_campaign_rate"] = round(float(df["AnyAccepted"].mean() * 100), 2)
    overall_rates = {k: float(v) for k, v in overall_rates.items()}

    result: dict[str, Any] = {
        "overall_acceptance_rate_pct": overall_rates,
        "segment_by": segment_by,
        "normalized_marital_status": normalize,
    }

    if segment_by != "none":
        grp = df.groupby(segment_by)[CAMPAIGN_COLS].mean().mul(100).round(2)
        result["by_segment"] = records_from_df(grp)

    return result


async def handle_channel_analysis(args: dict) -> dict:
    segment_by = args.get("segment_by", "none")

    df = DF.copy()
    df["has_children"] = ((df["Kidhome"] + df["Teenhome"]) > 0).map(
        {True: "Has children", False: "No children"}
    )
    df["TotalPurchases"] = df[PURCHASE_COLS].sum(axis=1)

    channel_totals = df[PURCHASE_COLS].sum()
    channel_share = (channel_totals / channel_totals.sum() * 100).round(2).to_dict()
    channel_avg = df[PURCHASE_COLS + ["NumWebVisitsMonth"]].mean().round(2).to_dict()

    result: dict[str, Any] = {
        "overall_channel_share_pct": {k: float(v) for k, v in channel_share.items()},
        "avg_purchases_per_customer": {k: float(v) for k, v in channel_avg.items()},
        "segment_by": segment_by,
    }

    if segment_by != "none":
        grp_col = "has_children" if segment_by == "has_children" else segment_by
        grp = df.groupby(grp_col)[PURCHASE_COLS + ["NumWebVisitsMonth"]].mean().round(2)
        result["by_segment"] = records_from_df(grp)

    return result


async def handle_rfm_segments(args: dict) -> dict:
    n_tiers = int(args.get("n_tiers", 3))

    df = DF.copy()
    df["Monetary"] = df[SPEND_COLS].sum(axis=1)
    df["Frequency"] = df[PURCHASE_COLS].sum(axis=1)

    # Recency: lower days = more recent = better → invert labels
    df["R_Score"] = pd.qcut(
        df["Recency"], n_tiers, labels=range(n_tiers, 0, -1)
    )
    # Use rank to break ties before qcut (many customers share same Frequency/Monetary values)
    df["F_Score"] = pd.qcut(
        df["Frequency"].rank(method="first"), n_tiers, labels=range(1, n_tiers + 1)
    )
    df["M_Score"] = pd.qcut(
        df["Monetary"].rank(method="first"), n_tiers, labels=range(1, n_tiers + 1)
    )
    df["RFM_Label"] = (
        df["R_Score"].astype(str)
        + df["F_Score"].astype(str)
        + df["M_Score"].astype(str)
    )

    grp = df.groupby("RFM_Label").agg(
        customer_count=("ID", "count"),
        avg_recency_days=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
        avg_income=("Income", "mean"),
        response_rate=("Response", "mean"),
    ).round(3)

    summary = records_from_df(grp)
    for row in summary:
        rr = row.pop("response_rate", None)
        row["response_rate_pct"] = round(float(rr) * 100, 2) if rr is not None else None

    return {
        "n_tiers": n_tiers,
        "scoring_note": (
            "R: higher score = purchased more recently; "
            "F: higher score = more total purchases; "
            "M: higher score = more total spend"
        ),
        "summary": sorted(summary, key=lambda x: x["RFM_Label"]),
    }


async def handle_income_spend_correlation(args: dict) -> dict:
    n_brackets = int(args.get("income_brackets", 5))
    exclude_outliers = args.get("exclude_income_outliers", True)

    df = DF.dropna(subset=["Income"]).copy()
    excluded_count = 0
    if exclude_outliers:
        mask = df["Income"] > 150_000
        excluded_count = int(mask.sum())
        df = df[~mask]

    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["IncomeBracket"] = pd.cut(df["Income"], bins=n_brackets)

    corr = float(df["Income"].corr(df["TotalSpend"]))

    agg_dict: dict[str, Any] = {
        "customer_count": ("Income", "count"),
        "avg_income": ("Income", "mean"),
        "avg_total_spend": ("TotalSpend", "mean"),
    }
    for col in SPEND_COLS:
        agg_dict[f"avg_{col.lower()}"] = (col, "mean")

    grp = df.groupby("IncomeBracket", observed=False).agg(**agg_dict).round(2)

    return {
        "income_spend_correlation": round(corr, 4),
        "excluded_outliers": exclude_outliers,
        "outliers_excluded_count": excluded_count,
        "customers_analyzed": len(df),
        "brackets": records_from_df(grp),
    }


async def handle_customer_tenure_analysis(args: dict) -> dict:
    cohort_months = int(args.get("cohort_months", 6))

    df = DF.copy()
    # Anchor to the latest registration date in the dataset for reproducibility
    reference_date = df["Dt_Customer"].max()
    df["TenureMonths"] = ((reference_date - df["Dt_Customer"]).dt.days / 30.44).astype(int)
    df["CohortStart"] = (df["TenureMonths"] // cohort_months) * cohort_months
    df["CohortLabel"] = df["CohortStart"].apply(
        lambda x: f"{x}-{x + cohort_months}mo"
    )
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)

    grp = df.groupby("CohortLabel").agg(
        customer_count=("ID", "count"),
        avg_total_spend=("TotalSpend", "mean"),
        avg_recency_days=("Recency", "mean"),
        response_rate=("Response", "mean"),
        avg_income=("Income", "mean"),
    ).round(3)

    cohorts = records_from_df(grp)
    for row in cohorts:
        rr = row.pop("response_rate", None)
        row["response_rate_pct"] = round(float(rr) * 100, 2) if rr is not None else None

    def cohort_sort_key(row: dict) -> int:
        label = row.get("CohortLabel", "0-0mo")
        try:
            return int(label.split("-")[0])
        except (ValueError, IndexError):
            return 0

    return {
        "reference_date": str(reference_date.date()),
        "cohort_months": cohort_months,
        "cohorts": sorted(cohorts, key=cohort_sort_key),
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
