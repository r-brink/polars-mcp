# pandas to Polars Translation Guide (LLM-Optimized)

## Core Principles

### 1. Always Use Lazy API

The lazy API allows Polars to optimize your query and get the best performance.

```python
# pandas (always eager)
df = pd.read_csv("data.csv")
result = df[df["age"] > 30].groupby("dept")["salary"].mean()

# Polars (lazy - PREFERRED)
result = (
    pl.scan_csv("data.csv")
    .filter(pl.col("age") > 30)
    .group_by("dept")
    .agg(pl.col("salary").mean())
    .collect()
)
```

### 2. Expression-Based vs Method Chaining

- **pandas**: Method chaining on DataFrames
- **Polars**: Expression-based with `pl.col()` in contexts

### 3. Performance Keys

- Use lazy evaluation (`.lazy()` → `.collect()`)
- Filter early (predicate pushdown)
- Select only needed columns (projection pushdown)
- Avoid `.map_elements()` - stay in expression API
- Generate all transformations at once (no loops)

---

## Reading Data

| pandas                            | Polars (Eager)                    | Polars (Lazy - PREFERRED)         |
| --------------------------------- | --------------------------------- | --------------------------------- |
| `pd.read_csv("file.csv")`         | `pl.read_csv("file.csv")`         | `pl.scan_csv("file.csv")`         |
| `pd.read_parquet("file.parquet")` | `pl.read_parquet("file.parquet")` | `pl.scan_parquet("file.parquet")` |
| `pd.read_excel("file.xlsx")`      | `pl.read_excel("file.xlsx")`      | N/A                               |
| `pd.read_json("file.json")`       | `pl.read_json("file.json")`       | `pl.scan_ndjson("file.ndjson")`   |

```python
# pandas
df = pd.read_csv("data.csv", usecols=["name", "age"])

# Polars (lazy)
lf = pl.scan_csv("data.csv").select(["name", "age"])
df = lf.collect()
```

---

## Column Selection

| Operation        | pandas                                | Polars                                  |
| ---------------- | ------------------------------------- | --------------------------------------- |
| Single column    | `df["col"]` or `df.col`               | `df.select("col")` or `df["col"]`       |
| Multiple columns | `df[["a", "b"]]`                      | `df.select(["a", "b"])`                 |
| All columns      | `df` or `df.loc[:, :]`                | `df.select(pl.all())`                   |
| Column range     | `df.loc[:, "a":"c"]`                  | `df.select(pl.col(""^a\|b\|c$""))`      |
| By dtype         | `df.select_dtypes(include="float64")` | `df.select(pl.col(pl.Float64))`         |
| Regex pattern    | `df.filter(regex="^sales_")`          | `df.select(pl.col("^sales_.*$"))`       |
| Exclude columns  | `df.drop(columns=["a", "b"])`         | `df.select(pl.all().exclude("a", "b"))` |

```python
# pandas
result = df[["name", "age", "salary"]]

# Polars
result = df.select(["name", "age", "salary"])
# or
result = df.select(pl.col("name", "age", "salary"))
```

---

## Creating/Modifying Columns

| Operation        | pandas                              | Polars                                                             |
| ---------------- | ----------------------------------- | ------------------------------------------------------------------ |
| Add column       | `df["new"] = df["a"] + df["b"]`     | `df.with_columns((pl.col("a") + pl.col("b")).alias("new"))`        |
| Multiple columns | Multiple assignments                | `df.with_columns([expr1, expr2, ...])`                             |
| Conditional      | `df["new"] = np.where(cond, x, y)`  | `df.with_columns(pl.when(cond).then(x).otherwise(y).alias("new"))` |
| Apply function   | `df["new"] = df["col"].apply(func)` | Avoid - use expressions                                            |

```python
# pandas
df["total"] = df["quantity"] * df["price"]
df["year"] = df["date"].dt.year
df["is_high"] = df["value"] > 100

# Polars
df = df.with_columns([
    (pl.col("quantity") * pl.col("price")).alias("total"),
    pl.col("date").dt.year().alias("year"),
    (pl.col("value") > 100).alias("is_high")
])
```

---

## Filtering Rows

| Operation        | pandas                                        | Polars                                                       |
| ---------------- | --------------------------------------------- | ------------------------------------------------------------ |
| Single condition | `df[df["age"] > 30]`                          | `df.filter(pl.col("age") > 30)`                              |
| Multiple AND     | `df[(df["age"] > 30) & (df["city"] == "NY")]` | `df.filter((pl.col("age") > 30) & (pl.col("city") == "NY"))` |
| Multiple OR      | `df[(df["age"] > 30) \| (df["age"] < 18)]`    | `df.filter((pl.col("age") > 30) \| (pl.col("age") < 18))`    |
| isin             | `df[df["city"].isin(["NY", "LA"])]`           | `df.filter(pl.col("city").is_in(["NY", "LA"]))`              |
| String contains  | `df[df["name"].str.contains("John")]`         | `df.filter(pl.col("name").str.contains("John"))`             |
| Not null         | `df[df["col"].notna()]`                       | `df.filter(pl.col("col").is_not_null())`                     |

```python
# pandas
result = df[(df["age"] > 25) & (df["salary"] > 50000) & (df["dept"] == "Sales")]

# Polars
result = df.filter(
    (pl.col("age") > 25) &
    (pl.col("salary") > 50000) &
    (pl.col("dept") == "Sales")
)
```

---

## Sorting

| Operation        | pandas                                                | Polars                                          |
| ---------------- | ----------------------------------------------------- | ----------------------------------------------- |
| Single column    | `df.sort_values("col")`                               | `df.sort("col")`                                |
| Descending       | `df.sort_values("col", ascending=False)`              | `df.sort("col", descending=True)`               |
| Multiple columns | `df.sort_values(["a", "b"])`                          | `df.sort(["a", "b"])`                           |
| Mixed order      | `df.sort_values(["a", "b"], ascending=[True, False])` | `df.sort(["a", "b"], descending=[False, True])` |

```python
# pandas
df = df.sort_values(["dept", "salary"], ascending=[True, False])

# Polars
df = df.sort(["dept", "salary"], descending=[False, True])
```

---

## Aggregation (GroupBy)

| Operation        | pandas                                                     | Polars                                                                       |
| ---------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Single agg       | `df.groupby("dept")["salary"].mean()`                      | `df.group_by("dept").agg(pl.col("salary").mean())`                           |
| Multiple aggs    | `df.groupby("dept")["salary"].agg(["mean", "sum"])`        | `df.group_by("dept").agg([pl.col("salary").mean(), pl.col("salary").sum()])` |
| Multiple columns | `df.groupby("dept").agg({"salary": "mean", "age": "max"})` | `df.group_by("dept").agg([pl.col("salary").mean(), pl.col("age").max()])`    |
| Count            | `df.groupby("dept").size()`                                | `df.group_by("dept").agg(pl.len())`                                          |
| Multiple groups  | `df.groupby(["dept", "city"])["salary"].mean()`            | `df.group_by(["dept", "city"]).agg(pl.col("salary").mean())`                 |

```python
# pandas
result = df.groupby("category").agg({
    "revenue": ["sum", "mean"],
    "quantity": "sum",
    "customer_id": "nunique"
})

# Polars
result = df.group_by("category").agg([
    pl.col("revenue").sum().alias("revenue_sum"),
    pl.col("revenue").mean().alias("revenue_mean"),
    pl.col("quantity").sum().alias("quantity_sum"),
    pl.col("customer_id").n_unique().alias("customer_count")
])
```

### Aggregation Functions

| pandas       | Polars                   |
| ------------ | ------------------------ |
| `.mean()`    | `.mean()`                |
| `.sum()`     | `.sum()`                 |
| `.min()`     | `.min()`                 |
| `.max()`     | `.max()`                 |
| `.std()`     | `.std()`                 |
| `.var()`     | `.var()`                 |
| `.median()`  | `.median()`              |
| `.count()`   | `.count()` or `pl.len()` |
| `.nunique()` | `.n_unique()`            |
| `.first()`   | `.first()`               |
| `.last()`    | `.last()`                |

---

## Window Functions / GroupBy + Transform

| Operation              | pandas                                           | Polars                                             |
| ---------------------- | ------------------------------------------------ | -------------------------------------------------- |
| Group mean (broadcast) | `df.groupby("dept")["salary"].transform("mean")` | `df.select(pl.col("salary").mean().over("dept"))`  |
| Rank within group      | `df.groupby("dept")["salary"].rank()`            | `df.select(pl.col("salary").rank().over("dept"))`  |
| Cumulative sum         | `df.groupby("id")["amount"].cumsum()`            | `df.select(pl.col("amount").cum_sum().over("id"))` |
| Shift within group     | `df.groupby("id")["value"].shift(1)`             | `df.select(pl.col("value").shift(1).over("id"))`   |

```python
# pandas - add group average to each row
df["dept_avg_salary"] = df.groupby("dept")["salary"].transform("mean")

# Polars - same result
df = df.with_columns(
    pl.col("salary").mean().over("dept").alias("dept_avg_salary")
)
```

**Key Difference:**

- pandas: `.groupby().transform()` broadcasts back to original rows
- Polars: `.over()` achieves the same (window function)

---

## Joins

| Operation                                       | pandas                                      | Polars                                     |
| ----------------------------------------------- | ------------------------------------------- | ------------------------------------------ |
| Inner join                                      | `df1.merge(df2, on="id")`                   | `df1.join(df2, on="id", how="inner")`      |
| Left join                                       | `df1.merge(df2, on="id", how="left")`       | `df1.join(df2, on="id", how="left")`       |
| Right join                                      | `df1.merge(df2, on="id", how="right")`      | `df1.join(df2, on="id", how="right")`      |
| Outer join                                      | `df1.merge(df2, on="id", how="outer")`      | `df1.join(df2, on="id", how="full")`       |
| Different keys                                  | `df1.merge(df2, left_on="a", right_on="b")` | `df1.join(df2, left_on="a", right_on="b")` |
| Semi-join (rows in df1 where key exists in df2) | `df1[df1['key'].isin(df2['key'])]`          | `df1.join(df2, on="key", how="semi")`      |
| Anti-join (rows in df1 where key NOT in df2)    | `df1[~df1['key'].isin(df2['key'])]`         | `df1.join(df2, on="key", how="anti")`      |

```python
# pandas
result = df1.merge(df2, on="customer_id", how="left")

# Polars
result = df1.join(df2, on="customer_id", how="left")

# pandas - Semi-join (keep orders where customer exists in active_customers)
active_order_ids = active_customers["customer_id"].unique()
orders_from_active = orders[orders["customer_id"].isin(active_order_ids)]

# Polars - Semi-join (more efficient)
orders_from_active = orders.join(active_customers, on="customer_id", how="semi")

# pandas - Anti-join (keep orders where customer NOT in active_customers)
orders_from_inactive = orders[~orders["customer_id"].isin(active_order_ids)]

# Polars - Anti-join (more efficient)
orders_from_inactive = orders.join(active_customers, on="customer_id", how="anti")
```

---

## Concatenation

| Operation         | pandas                          | Polars                                    |
| ----------------- | ------------------------------- | ----------------------------------------- |
| Vertical (rows)   | `pd.concat([df1, df2])`         | `pl.concat([df1, df2])`                   |
| Horizontal (cols) | `pd.concat([df1, df2], axis=1)` | `pl.concat([df1, df2], how="horizontal")` |

```python
# pandas - stack dataframes vertically
combined = pd.concat([df1, df2, df3], ignore_index=True)

# Polars
combined = pl.concat([df1, df2, df3])
```

---

## Handling Missing Values

| Operation      | pandas                      | Polars                                   |
| -------------- | --------------------------- | ---------------------------------------- |
| Check null     | `df["col"].isna()`          | `df.select(pl.col("col").is_null())`     |
| Check not null | `df["col"].notna()`         | `df.select(pl.col("col").is_not_null())` |
| Drop nulls     | `df.dropna()`               | `df.drop_nulls()`                        |
| Fill nulls     | `df.fillna(0)`              | `df.fill_null(0)`                        |
| Fill forward   | `df.fillna(method="ffill")` | `df.fill_null(strategy="forward")`       |
| Fill backward  | `df.fillna(method="bfill")` | `df.fill_null(strategy="backward")`      |

```python
# pandas
df["value"] = df["value"].fillna(0)
df["category"] = df["category"].fillna("Unknown")

# Polars
df = df.with_columns([
    pl.col("value").fill_null(0),
    pl.col("category").fill_null("Unknown")
])
```

---

## String Operations

| Operation | pandas                                | Polars                                               |
| --------- | ------------------------------------- | ---------------------------------------------------- |
| Uppercase | `df["col"].str.upper()`               | `df.select(pl.col("col").str.to_uppercase())`        |
| Lowercase | `df["col"].str.lower()`               | `df.select(pl.col("col").str.to_lowercase())`        |
| Strip     | `df["col"].str.strip()`               | `df.select(pl.col("col").str.strip_chars())`         |
| Contains  | `df["col"].str.contains("pattern")`   | `df.select(pl.col("col").str.contains("pattern"))`   |
| Replace   | `df["col"].str.replace("old", "new")` | `df.select(pl.col("col").str.replace("old", "new"))` |
| Split     | `df["col"].str.split(",")`            | `df.select(pl.col("col").str.split(","))`            |
| Length    | `df["col"].str.len()`                 | `df.select(pl.col("col").str.lengths())`             |

```python
# pandas
df["name_upper"] = df["name"].str.upper()
df["has_keyword"] = df["text"].str.contains("python", case=False)

# Polars
df = df.with_columns([
    pl.col("name").str.to_uppercase().alias("name_upper"),
    pl.col("text").str.contains("python").alias("has_keyword")
])
```

---

## Date/Time Operations

| Operation       | pandas                              | Polars                                            |
| --------------- | ----------------------------------- | ------------------------------------------------- |
| Extract year    | `df["date"].dt.year`                | `df.select(pl.col("date").dt.year())`             |
| Extract month   | `df["date"].dt.month`               | `df.select(pl.col("date").dt.month())`            |
| Extract day     | `df["date"].dt.day`                 | `df.select(pl.col("date").dt.day())`              |
| Extract weekday | `df["date"].dt.dayofweek`           | `df.select(pl.col("date").dt.weekday())`          |
| Date arithmetic | `df["date"] + pd.Timedelta(days=7)` | `df.select(pl.col("date") + pl.duration(days=7))` |
| Truncate        | `df["datetime"].dt.floor("D")`      | `df.select(pl.col("datetime").dt.truncate("1d"))` |

```python
# pandas
df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["is_weekend"] = df["order_date"].dt.dayofweek >= 5

# Polars
df = df.select([
    pl.col("order_date").dt.year().alias("year"),
    pl.col("order_date").dt.month().alias("month"),
    (pl.col("order_date").dt.weekday() >= 5).alias("is_weekend")
])
```

---

## Pivot and Melt

| Operation | pandas                                                     | Polars                                               |
| --------- | ---------------------------------------------------------- | ---------------------------------------------------- |
| Pivot     | `df.pivot_table(values="val", index="row", columns="col")` | `df.pivot(values="val", index="row", columns="col")` |
| Melt      | `df.melt(id_vars=["id"], value_vars=["a", "b"])`           | `df.melt(id_vars=["id"], value_vars=["a", "b"])`     |

```python
# pandas
pivoted = df.pivot_table(
    values="sales",
    index="region",
    columns="month",
    aggfunc="sum"
)

# Polars
pivoted = df.pivot(
    values="sales",
    index="region",
    columns="month",
    aggregate_function="sum"
)
```

---

## Unique and Duplicates

| Operation       | pandas                                        | Polars                                |
| --------------- | --------------------------------------------- | ------------------------------------- |
| Unique values   | `df["col"].unique()`                          | `df.select(pl.col("col").unique())`   |
| Count unique    | `df["col"].nunique()`                         | `df.select(pl.col("col").n_unique())` |
| Drop duplicates | `df.drop_duplicates()`                        | `df.unique()`                         |
| Drop by subset  | `df.drop_duplicates(subset=["col1", "col2"])` | `df.unique(subset=["col1", "col2"])`  |
| Keep first/last | `df.drop_duplicates(keep="first")`            | `df.unique(keep="first")`             |

```python
# pandas
unique_customers = df["customer_id"].nunique()
df_dedup = df.drop_duplicates(subset=["order_id"], keep="first")

# Polars
unique_customers = df.select(pl.col("customer_id").n_unique())
df_dedup = df.unique(subset=["order_id"], keep="first")
```

---

## Row Operations

| Operation       | pandas                      | Polars                    |
| --------------- | --------------------------- | ------------------------- |
| Head            | `df.head(10)`               | `df.head(10)`             |
| Tail            | `df.tail(10)`               | `df.tail(10)`             |
| Sample          | `df.sample(n=100)`          | `df.sample(n=100)`        |
| Sample fraction | `df.sample(frac=0.1)`       | `df.sample(fraction=0.1)` |
| Reset index     | `df.reset_index(drop=True)` | N/A (no index)            |

---

## Type Casting

| Operation    | pandas                        | Polars                                                       |
| ------------ | ----------------------------- | ------------------------------------------------------------ |
| Convert type | `df["col"].astype("float64")` | `df.select(pl.col("col").cast(pl.Float64))`                  |
| To datetime  | `pd.to_datetime(df["col"])`   | `df.select(pl.col("col").str.strptime(pl.Date, "%Y-%m-%d"))` |
| To string    | `df["col"].astype(str)`       | `df.select(pl.col("col").cast(pl.Utf8))`                     |

```python
# pandas
df["price"] = df["price"].astype("float64")
df["date"] = pd.to_datetime(df["date"])
df["id"] = df["id"].astype(str)

# Polars
df = df.select([
    pl.col("price").cast(pl.Float64),
    pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
    pl.col("id").cast(pl.Utf8)
])
```

---

## Common Anti-Patterns to Avoid

### Don't Use Python Loops

```python
# BAD
for col in ["a", "b", "c"]:
    df = df.with_columns(pl.col(col) * 2)

# GOOD
df = df.with_columns([pl.col(col) * 2 for col in ["a", "b", "c"]])
# or
df = df.with_columns(pl.col("a", "b", "c") * 2)
```

### Don't Use .map_elements() (pandas .apply())

```python
# BAD (slow, loses parallelization)
df.select(pl.col("value").map_elements(lambda x: x * 2))

# GOOD
df.select(pl.col("value") * 2)
```

### Don't Use Eager When Lazy Available

```python
# BAD
df = pl.read_csv("large.csv")
result = df.filter(pl.col("year") == 2024).group_by("dept").agg(pl.col("salary").mean())

# GOOD
result = (
    pl.scan_csv("large.csv")
    .filter(pl.col("year") == 2024)
    .group_by("dept")
    .agg(pl.col("salary").mean())
    .collect()
)
```

### Don't Filter After Expensive Operations

```python
# BAD
df.group_by("id").agg(expensive_expr()).filter(pl.col("year") == 2024)

# GOOD
df.filter(pl.col("year") == 2024).group_by("id").agg(expensive_expr())
```

---

## Complete Example: pandas vs Polars

### pandas

```python
import pandas as pd

# Read data
df = pd.read_csv("sales.csv")

# Filter
df = df[df["year"] == 2024]

# Add computed columns
df["profit"] = df["revenue"] - df["cost"]
df["margin"] = (df["profit"] / df["revenue"]) * 100

# Group and aggregate
result = df.groupby(["region", "category"]).agg({
    "profit": ["sum", "mean"],
    "customer_id": "nunique",
    "order_id": "count"
})

# Flatten column names
result.columns = ["_".join(col) for col in result.columns]
result = result.reset_index()

# Filter groups
result = result[result["profit_sum"] > 10000]

# Sort
result = result.sort_values("profit_sum", ascending=False)
```

### Polars (Optimized)

```python
import polars as pl

result = (
    pl.scan_csv("sales.csv")
    # Filter early (predicate pushdown)
    .filter(pl.col("year") == 2024)
    # Add computed columns
    .with_columns([
        (pl.col("revenue") - pl.col("cost")).alias("profit"),
        ((pl.col("revenue") - pl.col("cost")) / pl.col("revenue") * 100).alias("margin")
    ])
    # Group and aggregate
    .group_by(["region", "category"])
    .agg([
        pl.col("profit").sum().alias("profit_sum"),
        pl.col("profit").mean().alias("profit_mean"),
        pl.col("customer_id").n_unique().alias("customer_id_nunique"),
        pl.col("order_id").count().alias("order_id_count")
    ])
    # Filter groups
    .filter(pl.col("profit_sum") > 10000)
    # Sort
    .sort("profit_sum", descending=True)
    # Execute
    .collect()
)
```

---

## Quick Reference: Key Differences

| Aspect               | pandas                   | Polars                                    |
| -------------------- | ------------------------ | ----------------------------------------- |
| **Evaluation**       | Eager                    | Lazy (preferred)                          |
| **Column selection** | `df["col"]`              | `pl.col("col")`                           |
| **Multiple columns** | `df[["a", "b"]]`         | `df.select(["a", "b"])`                   |
| **Add columns**      | Assignment               | `.with_columns()`                         |
| **Filter**           | `df[condition]`          | `.filter(condition)`                      |
| **Group by**         | `.groupby().agg()`       | `.group_by().agg()`                       |
| **Window**           | `.groupby().transform()` | `.over()`                                 |
| **Apply function**   | `.apply(func)`           | Use expressions (avoid `.map_elements()`) |
| **Null check**       | `.isna()`                | `.is_null()`                              |
| **String upper**     | `.str.upper()`           | `.str.to_uppercase()`                     |
| **Unique count**     | `.nunique()`             | `.n_unique()`                             |

---

## Performance Optimization Checklist

- Do use `scan_csv` / `scan_parquet` instead of `read_csv` / `read_parquet`
- Do use `.lazy()` and `.collect()` pattern
- Do use filter early in the query chain
- Do use select only needed columns early
- Do avoid `.map_elements()` - use native expressions
- Do generate all column transformations in one `.with_columns()` call
- Do use `.over()` for window functions instead of explicit joins
- Do let Polars handle optimization (trust predicate/projection pushdown)

---

## Common Pitfalls

1. **Using aggregations in `.with_columns()`**: Use `.over()` instead
2. **Forgetting `.collect()`**: Lazy queries need `.collect()` to execute
3. **Using Python loops**: Generate list of expressions instead
4. **Not using `.alias()`**: Always name computed columns
5. **Mixing eager and lazy**: Start with lazy, stay lazy until `.collect()`
