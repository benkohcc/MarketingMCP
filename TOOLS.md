# MCP Tool Reference

Detailed explanation of each tool exposed by the Marketing Campaign MCP server.

---

## How the server works (before the tools)

**`load_data()`** — Runs once when the server starts. Reads `marketing_campaign.csv` using `;` as the delimiter (it's not a standard comma-separated file), strips the invisible BOM character from the first column using `utf-8-sig` encoding, and parses `Dt_Customer` as a proper date. The result is stored as `DF` — a single shared DataFrame that all tools read from.

**`normalize_marital()`** — A helper that makes a copy of the DataFrame and replaces the junk `Marital_Status` values (`Alone`, `YOLO`, `Absurd`) with `Single`. Called by tools that segment by marital status when the `normalize_marital_status` flag is true (which it is by default).

**`records_from_df()`** — A helper that converts a pandas DataFrame into a plain list of Python dicts that can be safely serialised to JSON. The main problem it solves is that pandas uses its own numeric types (numpy int64, float64) that the standard `json` module can't handle — this function casts them to native Python types and replaces `NaN`/`Inf` with `None`.

---

## The 7 Tools

### `get_dataset_overview`
**No inputs required.**

Iterates through every column in the DataFrame and builds a profile of it:
- **Numeric columns** (Income, MntWines, etc.): reports min, max, mean, and count of missing values
- **Date columns** (Dt_Customer): reports the earliest and latest dates
- **Categorical columns** (Education, Marital_Status): reports how many unique values exist, and lists them all if there are 12 or fewer

Also appends two hardcoded data quality warnings — the 24 missing Income rows, and the anomalous marital status values. This is designed to be called first so the agent understands the shape of the data before doing any analysis.

---

### `get_spending_by_segment`
**Inputs:** `segment_by` (Education or Marital_Status), `normalize_marital_status` (bool, default true)

Adds up each customer's spend across all 6 product categories (Wines, Fruits, Meat, Fish, Sweets, Gold) into a `TotalSpend` column, then groups all customers by the chosen demographic dimension and computes the **average spend per product per group**. Returns the segments sorted from highest to lowest total spend, so the most valuable segments appear first.

The key pandas operation is `groupby(segment_by)[spend_cols].mean()` — it collapses 2,239 rows into one row per segment, each containing the average spend figure for that group across every product.

---

### `get_campaign_performance`
**Inputs:** `segment_by` (Education, Marital_Status, or none), `normalize_marital_status` (bool, default true)

Computes what percentage of customers accepted each of the 6 campaigns. It does this by taking the mean of each binary campaign column (0/1) and multiplying by 100 — since the mean of a 0/1 column is just the proportion of 1s, this gives you the acceptance rate directly.

Also derives an `AnyAccepted` column (true if a customer accepted *at least one* campaign) to give an overall reach figure. If a `segment_by` dimension is provided, it runs the same mean calculation per group using `groupby`, giving acceptance rates broken down by e.g. education level.

---

### `get_channel_analysis`
**Inputs:** `segment_by` (Education, Marital_Status, has_children, or none)

Looks at the four purchase channels — web, catalog, store, and deal purchases — and answers two questions:
1. **Channel share**: of all purchases across all customers, what percentage went through each channel? (Sums each channel across all customers, then divides by the grand total.)
2. **Average per customer**: how many purchases does a typical customer make through each channel per period?

The `has_children` segmentation option is computed on the fly by checking whether `Kidhome + Teenhome > 0`, creating a derived binary group not present in the raw data. When a segment is specified, it re-runs the averages via `groupby` to show how channel behaviour differs between groups.

---

### `get_rfm_segments`
**Inputs:** `n_tiers` (2–5, default 3)

Implements RFM (Recency, Frequency, Monetary) scoring — a standard marketing technique for ranking customers by their value and engagement.

- **Monetary**: sum of all 6 spend columns per customer
- **Frequency**: sum of all 4 purchase channel columns per customer
- **Recency**: already in the data as days since last purchase

Each dimension is then split into `n_tiers` equal-sized buckets using `pd.qcut` (quantile-based cutting, so each bucket has the same number of customers). Recency scores are **inverted** — a lower recency (purchased more recently) gets a *higher* score — which is the standard RFM convention. Frequency and Monetary use `.rank(method='first')` before cutting to handle the large number of tied values (e.g. many customers with £0 fruit spend).

Each customer ends up with a 3-digit label like `"333"` (best) or `"111"` (worst). The tool then groups by this label and reports the average characteristics and campaign response rate of each segment.

---

### `get_income_spend_correlation`
**Inputs:** `income_brackets` (3–10, default 5), `exclude_income_outliers` (bool, default true)

First drops the 24 rows with missing Income. Then optionally removes customers with income above £150,000 — these are almost certainly data entry errors (one entry is 666,666) that would heavily distort the brackets.

Computes the **Pearson correlation coefficient** between Income and TotalSpend — a number between -1 and 1 measuring how linearly related they are. In this dataset it's 0.82, indicating a strong positive relationship.

Then uses `pd.cut` (equal-width brackets, unlike `pd.qcut` which is equal-count) to divide customers into income bands and computes average spend per product category within each band. This lets the agent see not just that income predicts spend, but *where* the spend profile changes as income rises.

---

### `get_customer_tenure_analysis`
**Inputs:** `cohort_months` (1–12, default 6)

Calculates how long each customer has been registered by subtracting their `Dt_Customer` date from the **most recent registration date in the dataset** (2014-06-29). This is used as the anchor rather than today's date so the results are always consistent regardless of when you run it.

Tenure in days is converted to approximate months (`/ 30.44`), then customers are assigned to cohorts based on which `cohort_months`-wide window they fall into (e.g. 0–6mo, 6–12mo, 12–18mo, 18–24mo). For each cohort it reports average total spend, average recency, campaign response rate, and average income — letting the agent see whether longer-tenure customers behave differently from newer ones.
