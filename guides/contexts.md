# Expression Contexts in Polars

## Core Concepts

**Expressions** are lazy representations of data transformations. They don't compute until placed in a **context**.

**Contexts** are methods that execute expressions and determine their behavior:

- What columns appear in output
- How expressions interact with dataframe structure
- When broadcasting occurs

## The Four Main Contexts

### 1. `select()` - Column Projection

**Purpose:** Choose and transform columns. Only specified columns appear in output.

**Rules:**

- All expressions must produce series of **same length** OR scalars
- Scalars are broadcast to match series length
- Returns only the selected columns

**When to use:** Creating new columns, aggregations, or subsetting columns.

```python
# Basic selection
lf.select([
    pl.col("name"),
    pl.col("salary"),
])

# Computed columns
lf.select([
    pl.col("weight") / (pl.col("height") ** 2).alias("bmi"),
    pl.col("age").mean().alias("avg_age"),  # Aggregation (scalar broadcast)
    pl.lit(25).alias("target"),              # Literal (scalar broadcast)
])

# Complex expressions
lf.select([
    (pl.col("revenue") - pl.col("cost")).alias("profit"),
    ((pl.col("revenue") - pl.col("cost")) / pl.col("revenue") * 100).alias("margin_pct"),
])
```

---

### 2. `with_columns()` - Column Addition/Mutation

**Purpose:** Add or replace columns while keeping all original columns.

**Rules:**

- All expressions must produce series of **same length as original dataframe**
- Cannot use aggregations that reduce to scalar (use `select()` for that)
- Original columns preserved unless explicitly overwritten

**When to use:** Adding calculated columns, type casting, feature engineering.

```python
# Add new columns
lf.with_columns([
    (pl.col("quantity") * pl.col("price")).alias("total"),
    pl.col("date").dt.year().alias("year"),
])

# Replace existing column
lf.with_columns([
    pl.col("price").cast(pl.Float64),  # Overwrites "price"
])

# Multiple operations
lf.with_columns([
    pl.col("name").str.to_uppercase().alias("name_upper"),
    pl.col("salary").rank().alias("salary_rank"),
    (pl.col("bonus") / pl.col("salary")).alias("bonus_ratio"),
])
```

---

### 3. `filter()` - Row Filtering

**Purpose:** Keep rows where boolean expressions evaluate to `True`.

**Rules:**

- All expressions must evaluate to boolean type
- Multiple conditions combined with `&` (AND) or `|` (OR)
- Use parentheses for condition grouping

**When to use:** Subsetting rows based on conditions.

```python
# Single condition
lf.filter(pl.col("age") >= 18)

# Multiple conditions (AND)
lf.filter(
    (pl.col("age") >= 18) &
    (pl.col("country") == "US")
)

# Multiple conditions (OR)
lf.filter(
    (pl.col("status") == "active") |
    (pl.col("status") == "pending")
)

# Using is_in for multiple values
lf.filter(pl.col("status").is_in(["active", "pending"]))

# Complex conditions
lf.filter(
    (pl.col("date").is_between(start_date, end_date)) &
    (pl.col("amount") > 1000) &
    (pl.col("category").str.contains("sales"))
)

# Negation
lf.filter(~pl.col("is_deleted"))
lf.filter(pl.col("value").is_not_null())
```

---

### 4. `group_by()` + `agg()` - Grouped Aggregations

**Purpose:** Group rows by unique values and compute aggregations per group.

**Rules:**

- `group_by()` defines grouping columns (can be expressions)
- `agg()` defines aggregating expressions
- Output has one row per unique group
- Output columns = grouping columns + aggregation results

**When to use:** Computing statistics per category, group-wise transformations.

```python
# Simple grouping
lf.group_by("department").agg([
    pl.col("salary").mean().alias("avg_salary"),
    pl.col("employee_id").count().alias("headcount"),
])

# Multiple grouping columns
lf.group_by(["department", "location"]).agg([
    pl.col("salary").sum(),
    pl.col("salary").max(),
])

# Dynamic grouping (computed groups)
lf.group_by(
    (pl.col("date").dt.year()).alias("year")
).agg([
    pl.col("revenue").sum(),
])

# Multiple grouping expressions
lf.group_by([
    (pl.col("date").dt.year() // 10 * 10).alias("decade"),
    (pl.col("height") < 1.7).alias("is_short"),
]).agg([
    pl.len().alias("count"),
    pl.col("weight").mean().alias("avg_weight"),
])

# Maintain group order (don't sort)
lf.group_by("category", maintain_order=True).agg([
    pl.col("value").sum()
])
```

---

## Additional Contexts

### `sort()` - Row Ordering

```python
# Single column
lf.sort("date", descending=True)

# Multiple columns
lf.sort(["department", "salary"], descending=[False, True])

# By expression
lf.sort((pl.col("revenue") - pl.col("cost")), descending=True)

# Maintain null positions
lf.sort("value", nulls_last=True)
```

### `join()` - Combining DataFrames

```python
# Inner join
lf.join(other, on="id", how="inner")

# Left join
lf.join(other, on="user_id", how="left")

# Join on multiple columns
lf.join(other, on=["year", "month"], how="inner")

# Join with different column names
lf.join(other, left_on="id", right_on="user_id", how="left")
```

---

## Constructing Polars Queries

### Query Building Pattern

Follow this typical order:

```python
result = (
    lf
    # 1. Filter early (predicate pushdown)
    .filter(pl.col("year") == 2024)

    # 2. Add computed columns needed for later steps
    .with_columns([
        (pl.col("revenue") - pl.col("cost")).alias("profit")
    ])

    # 3. Group and aggregate
    .group_by("region")
    .agg([
        pl.col("profit").sum().alias("total_profit"),
        pl.col("profit").mean().alias("avg_profit"),
        pl.len().alias("count"),
    ])

    # 4. Filter groups
    .filter(pl.col("count") > 10)

    # 5. Sort results
    .sort("total_profit", descending=True)

    # 6. Select final columns
    .select([
        pl.col("region"),
        pl.col("total_profit"),
        pl.col("avg_profit"),
    ])

    # 7. Execute
    .collect()
)
```

### Context Selection Guide

| Task                | Context                | Example                                                  |
| ------------------- | ---------------------- | -------------------------------------------------------- |
| Subset columns      | `select()`             | `lf.select(["name", "age"])`                             |
| Add computed column | `with_columns()`       | `lf.with_columns(total=pl.col("qty") * pl.col("price"))` |
| Remove rows         | `filter()`             | `lf.filter(pl.col("age") >= 18)`                         |
| Aggregate data      | `group_by()` + `agg()` | `lf.group_by("dept").agg(pl.col("salary").mean())`       |
| Order rows          | `sort()`               | `lf.sort("date")`                                        |
| Combine tables      | `join()`               | `lf.join(other, on="id")`                                |

---

## Expression Expansion

Expressions can expand to multiple columns using selectors:

```python
# All numeric columns
lf.select(pl.col(pl.NUMERIC_DTYPES))

# All columns except some
lf.select(pl.all().exclude("id", "created_at"))

# Regex pattern matching
lf.select(pl.col("^sales_.*$"))

# Apply transformation to multiple columns
lf.select(pl.col("height", "weight").mean())  # Expands to 2 expressions

# Type-based selection with transformation
lf.select((pl.col(pl.Float64) * 1.1).name.suffix("_adjusted"))
```

**Expression expansion is parallel:** When an expression expands to multiple columns, Polars executes them in parallel.

---

## Broadcasting Rules

**Scalars are broadcast to match series length:**

```python
# In select: aggregation broadcast to all rows
lf.select([
    pl.col("value"),
    pl.col("value").mean().alias("avg"),  # Scalar broadcast
])

# Within expression: broadcast in operations
lf.select([
    # Mean is computed once, then broadcast for subtraction
    (pl.col("value") - pl.col("value").mean()).alias("deviation")
])
```

**In `with_columns()`, no scalar broadcasting from aggregations:**

```python
# This works: expression produces series of same length
lf.with_columns([
    pl.col("value").rank().alias("rank")
])

# This FAILS: mean() produces scalar, but with_columns needs series
# lf.with_columns([
#     pl.col("value").mean().alias("avg")  # ERROR
# ])

# Solution: use over() for window function
lf.with_columns([
    pl.col("value").mean().over("category").alias("category_avg")
])
```

---

## Common Patterns

### Pattern 1: Filter â†’ Group â†’ Filter Groups

```python
lf.filter(pl.col("status") == "active") \
  .group_by("category") \
  .agg(pl.col("value").sum().alias("total")) \
  .filter(pl.col("total") > 1000)
```

### Pattern 2: Add Computed Columns â†’ Filter on Them

```python
lf.with_columns([
    (pl.col("revenue") / pl.col("cost")).alias("roi")
]) \
  .filter(pl.col("roi") > 1.5)
```

### Pattern 3: Multiple Aggregations with Renaming

```python
lf.group_by("region").agg([
    pl.col("sales").sum().alias("total_sales"),
    pl.col("sales").mean().alias("avg_sales"),
    pl.col("sales").std().alias("std_sales"),
    pl.len().alias("num_transactions"),
])
```

### Pattern 4: Window Functions (group aggregation per row)

```python
lf.with_columns([
    pl.col("salary").mean().over("department").alias("dept_avg_salary"),
    pl.col("sales").rank().over("region").alias("regional_rank"),
])
```

### Pattern 5: Conditional Aggregation

```python
lf.group_by("category").agg([
    pl.col("amount").filter(pl.col("status") == "paid").sum().alias("paid_total"),
    pl.col("amount").filter(pl.col("status") == "pending").sum().alias("pending_total"),
])
```
