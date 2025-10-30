# Polars Expression Contexts: Complete Reference Guide

**Purpose**: Comprehensive guide to expression contexts in Polars, optimized for LLM understanding and code generation.

---

## Core Concept

Expressions in Polars behave differently based on their **context**. A context is the method that receives the expressions (`select()`, `with_columns()`, `filter()`, `group_by()` + `agg()`). The same expression can produce different results depending on where it's used.

---

## Context Types Overview

| Context            | Purpose                   | Keeps Existing Columns | Returns                    | Typical Use                                 |
| ------------------ | ------------------------- | ---------------------- | -------------------------- | ------------------------------------------- |
| `select()`         | Project/transform columns | No                     | Only specified columns     | Choose and compute specific outputs         |
| `with_columns()`   | Add/modify columns        | Yes                    | All columns + new/modified | Enrich dataframe with derived values        |
| `filter()`         | Subset rows               | Yes                    | Rows matching predicate    | Remove unwanted data                        |
| `group_by().agg()` | Aggregate by groups       | No                     | One row per group          | Summarize grouped data                      |
| `over()`           | Window functions          | Yes                    | Original row count         | Per-group calculations preserving structure |
| `sort()`           | Order rows                | Yes                    | All rows reordered         | Arrange data by column values               |

---

## 1. SELECT CONTEXT

**Behavior**: Transforms and selects columns. Only returns the specified columns.

### Basic Selection

```python
# Select specific columns
df.select("name", "age")

# Select and transform
df.select([
    pl.col("name"),
    pl.col("price") * 1.1,
    (pl.col("cost") + pl.col("tax")).alias("total")
])
```

### Expression Expansion

```python
# By data type - selects ALL columns matching the type
df.select(pl.col(pl.NUMERIC_DTYPES))  # All numeric columns
df.select(pl.col(pl.Float64, pl.Float32))  # Multiple types

# By pattern (regex: must start with ^ and end with $)
df.select(pl.col("^sales_.*$"))  # Columns starting with "sales_"
df.select(pl.col("^.*_total$"))  # Columns ending with "_total"

# Select all columns
df.select(pl.all())  # Equivalent to pl.col("*")

# Exclude columns
df.select(pl.all().exclude("id", "temp"))
df.select(pl.col(pl.NUMERIC_DTYPES).exclude("^temp_.*$"))
```

### Column Renaming in Select

```python
# Single column alias
df.select(pl.col("revenue").alias("total_revenue"))

# Prefix/suffix for expanded expressions
df.select(
    pl.col("^2024_.*$").name.prefix("year_"),
    pl.col(pl.Float64).name.suffix("_usd")
)

# Dynamic renaming
df.select(pl.all().name.map(str.upper))  # Uppercase all names
df.select(pl.all().name.to_lowercase())  # Lowercase all names
```

### Selectors (Python only)

```python
import polars.selectors as cs

# Select by data type patterns
df.select(cs.numeric())  # All numeric columns
df.select(cs.temporal())  # Date, Datetime, Duration
df.select(cs.string())   # String columns

# Select by name patterns
df.select(cs.starts_with("sales_"))
df.select(cs.ends_with("_total"))
df.select(cs.contains("temp"))
df.select(cs.matches(r"^\d+$"))  # Regex pattern

# Set operations
df.select(cs.numeric() | cs.temporal())  # Union
df.select(cs.numeric() & ~cs.contains("_temp"))  # Intersection & complement
df.select(cs.numeric() - cs.ends_with("_id"))  # Difference

# Chain expressions on selectors
df.select((cs.numeric() * 1.1).name.suffix("_inflated"))

# Use as_expr() to resolve operator ambiguity
df.select((~cs.starts_with("is_").as_expr()).name.prefix("not_"))
```

---

## 2. WITH_COLUMNS CONTEXT

**Behavior**: Adds or modifies columns while keeping all existing columns.

### Basic Usage

```python
# Add new columns
df.with_columns([
    (pl.col("price") * pl.col("quantity")).alias("revenue"),
    pl.col("date").dt.year().alias("year")
])

# Modify existing columns (overwrites)
df.with_columns([
    pl.col("price").cast(pl.Float64),  # Convert type
    pl.col("name").str.to_uppercase()   # Transform in place
])
```

### Multiple Transformations

```python
# Chain multiple operations
df.with_columns([
    pl.col("value").fill_null(0),
    (pl.col("amount") * 1.1).round(2),
    pl.when(pl.col("status") == "active")
      .then(1)
      .otherwise(0)
      .alias("is_active")
])
```

### Programmatic Column Generation

```python
# DON'T: Use loop with multiple with_columns calls
# for col in columns:
#     df = df.with_columns(pl.col(col) * 2)

# DO: Generate all expressions first, use once
def generate_scaled_columns(columns, scale):
    return [pl.col(col) * scale for col in columns]

df.with_columns(generate_scaled_columns(["a", "b", "c"], 2))

# Or use generators
def scaled_expr(columns, scale):
    for col in columns:
        yield (pl.col(col) * scale).alias(f"{col}_scaled")

df.with_columns(scaled_expr(["a", "b", "c"], 2))
```

---

## 3. FILTER CONTEXT

**Behavior**: Boolean expressions that determine which rows to keep.

### Basic Filtering

```python
# Single condition
df.filter(pl.col("age") >= 18)
df.filter(pl.col("status") == "active")

# Multiple conditions (AND)
df.filter(
    (pl.col("age") >= 18) &
    (pl.col("country") == "US") &
    (pl.col("score") > 80)
)

# Multiple conditions (OR)
df.filter(
    (pl.col("status") == "active") |
    (pl.col("status") == "pending")
)
```

### Operators Reference

```python
# Comparison operators
pl.col("x") > 5      # Greater than
pl.col("x") >= 5     # Greater than or equal
pl.col("x") < 5      # Less than
pl.col("x") <= 5     # Less than or equal
pl.col("x") == 5     # Equal
pl.col("x") != 5     # Not equal

# Boolean operators (use & | ~ not and/or/not)
(condition1) & (condition2)   # AND
(condition1) | (condition2)   # OR
~condition                     # NOT

# Membership tests
pl.col("status").is_in(["active", "pending"])
pl.col("category").is_in(valid_categories)

# Null checks
pl.col("value").is_null()
pl.col("value").is_not_null()

# String operations
pl.col("name").str.contains("test")
pl.col("name").str.starts_with("Mr")
pl.col("name").str.ends_with("son")

# Between (inclusive)
pl.col("age").is_between(18, 65)
```

### Complex Predicates

```python
# Nested conditions
df.filter(
    (
        (pl.col("region") == "EMEA") &
        (pl.col("revenue") > 1_000_000)
    ) | (
        (pl.col("region") == "APAC") &
        (pl.col("growth_rate") > 0.5)
    )
)

# Using when-then-otherwise in filter
df.filter(
    pl.when(pl.col("tier") == "premium")
      .then(pl.col("amount") > 1000)
      .otherwise(pl.col("amount") > 100)
)
```

### Important: Null Behavior

```python
# Nulls in comparisons result in null (falsy in filter context)
df.filter(pl.col("value") > 10)  # Excludes nulls
df.filter((pl.col("value") > 10) | pl.col("value").is_null())  # Include nulls
```

---

## 4. GROUP_BY + AGG CONTEXT

**Behavior**: Aggregates data by groups. Returns one row per unique group combination.

### Basic Aggregation

```python
# Single grouping column
df.group_by("category").agg([
    pl.col("revenue").sum(),
    pl.col("customer_id").n_unique().alias("unique_customers"),
    pl.col("order_id").count().alias("order_count")
])

# Multiple grouping columns
df.group_by(["region", "category"]).agg([
    pl.col("revenue").mean().alias("avg_revenue"),
    pl.col("quantity").sum()
])

# Maintain original order
df.group_by("category", maintain_order=True).agg(...)
```

### Common Aggregation Functions

```python
df.group_by("group").agg([
    # Statistical
    pl.col("value").sum(),
    pl.col("value").mean(),
    pl.col("value").median(),
    pl.col("value").std(),
    pl.col("value").var(),
    pl.col("value").min(),
    pl.col("value").max(),
    pl.col("value").quantile(0.95),

    # Counting
    pl.len(),  # Count rows in group (includes nulls)
    pl.col("id").count(),  # Count non-null values
    pl.col("id").n_unique(),  # Count distinct values
    pl.col("id").null_count(),  # Count null values

    # First/last
    pl.col("name").first(),
    pl.col("name").last(),

    # Collection
    pl.col("tags").flatten(),  # Concatenate lists
    pl.col("name").implode(),  # Collect into list (alias for agg_groups)

    # All values (creates list column)
    pl.col("score"),  # Shorthand for .implode()
])
```

### Conditional Aggregation

```python
# Filter within aggregation
df.group_by("department").agg([
    pl.col("salary").filter(pl.col("level") == "senior").mean().alias("avg_senior_salary"),
    (pl.col("status") == "active").sum().alias("active_count"),
    (pl.col("bonus") > 0).sum().alias("bonus_recipients")
])

# Multiple conditions
df.group_by("state").agg([
    (pl.col("party") == "Republican").sum().alias("republican_count"),
    (pl.col("party") == "Democrat").sum().alias("democrat_count")
])
```

### Sorting Within Groups

```python
# Get top N per group
df.group_by("category").agg([
    pl.col("product_name")
      .sort_by("revenue", descending=True)
      .first()
      .alias("top_product"),

    pl.col("revenue")
      .sort_by("revenue", descending=True)
      .first()
      .alias("highest_revenue")
])

# Sort by one column, get values from another
df.group_by("type").agg([
    pl.col("name").sort_by("score").first().alias("lowest_scorer")
])
```

### Using Helper Functions

```python
# Define reusable aggregation functions
def compute_age():
    return 2024 - pl.col("birth_year")

def stats_by_condition(column, condition_expr):
    return [
        column.filter(condition_expr).mean().alias(f"avg_{column.meta.output_name()}"),
        column.filter(condition_expr).count().alias(f"count_{column.meta.output_name()}")
    ]

df.group_by("department").agg([
    compute_age().mean().alias("avg_age"),
    *stats_by_condition(pl.col("salary"), pl.col("status") == "active")
])
```

### Performance Note

```python
# AVOID: Python lambda/UDF in aggregation (slow, breaks parallelization)
# df.group_by("x").agg(pl.col("y").map_elements(custom_func))

# PREFER: Stay within Polars expression API
df.group_by("x").agg([
    pl.col("y").sum(),
    pl.col("y").mean(),
    pl.when(pl.col("y") > 0).then(pl.col("y")).otherwise(0).sum()
])
```

---

## 5. WINDOW FUNCTIONS (OVER CONTEXT)

**Behavior**: Applies expressions over groups while maintaining the original dataframe structure. Returns same number of rows as input (usually).

### Basic Window Operations

```python
# Rank within groups
df.select([
    pl.col("name"),
    pl.col("category"),
    pl.col("score").rank("dense", descending=True).over("category").alias("rank_in_category")
])

# Aggregate value broadcasted to all rows in group
df.select([
    pl.col("name"),
    pl.col("department"),
    pl.col("salary"),
    pl.col("salary").mean().over("department").alias("dept_avg_salary"),
    (pl.col("salary") - pl.col("salary").mean().over("department")).alias("diff_from_avg")
])
```

### Multiple Grouping Columns

```python
# Window over multiple columns
df.select([
    pl.col("name"),
    pl.col("country"),
    pl.col("city"),
    pl.col("sales").sum().over(["country", "city"]).alias("city_total")
])
```

### Common Window Patterns

```python
df.select([
    pl.all(),

    # Ranking within groups
    pl.col("value").rank().over("group"),
    pl.col("value").rank("dense").over("group"),

    # Running aggregations
    pl.col("amount").cum_sum().over("customer_id"),
    pl.col("amount").cum_mean().over("customer_id"),

    # Row position functions
    pl.col("id").cum_count().over("group").alias("row_num"),

    # Statistical comparisons
    (pl.col("score") > pl.col("score").mean().over("class")).alias("above_class_avg"),

    # Shift operations
    pl.col("price").shift(1).over("product_id").alias("prev_price"),
    (pl.col("price") - pl.col("price").shift(1).over("product_id")).alias("price_change")
])
```

### Mapping Strategies

Window functions support different mapping strategies via `mapping_strategy` parameter:

```python
# 1. "group_to_rows" (DEFAULT): Results maintain original row positions
df.select(
    pl.col("athlete"),
    pl.col("country"),
    pl.col("athlete").sort_by("rank")
      .over("country", mapping_strategy="group_to_rows")
)
# Original row order preserved, values reordered within groups

# 2. "explode": Groups are kept together, faster but reorders rows
df.select(
    pl.col("athlete"),
    pl.col("country"),
    pl.col("athlete").sort_by("rank")
      .over("country", mapping_strategy="explode")
)
# Rows grouped together by country, original order not preserved

# 3. "join": Aggregates results into list, broadcasts to all group rows
df.select(
    pl.col("athlete"),
    pl.col("country"),
    pl.col("rank").sort()
      .over("country", mapping_strategy="join")
      .alias("all_ranks_in_country")
)
# Each row gets list of all values in group
```

### Window vs Group By

```python
# group_by: Returns one row per group
df.group_by("category").agg(pl.col("price").mean())
# Result: n_unique(category) rows

# over: Returns same number of rows, with aggregated value broadcasted
df.select(pl.col("price").mean().over("category"))
# Result: original row count

# They can produce similar results with explode:
df.group_by("category").agg(pl.col("price").mean()).explode("price")
# But rows may be in different order
```

### Advanced Window Examples

```python
# Top N per group with details
df.select([
    pl.col("category").head(3).over("category", mapping_strategy="explode"),
    pl.col("product")
      .sort_by("revenue", descending=True)
      .head(3)
      .over("category", mapping_strategy="explode"),
    pl.col("revenue")
      .sort_by("revenue", descending=True)
      .head(3)
      .over("category", mapping_strategy="explode")
])

# Multiple window operations
df.select([
    pl.all(),
    pl.col("speed").rank("dense", descending=True).over("type").alias("speed_rank"),
    pl.col("name").sort_by("speed", descending=True).head(3).over("type", mapping_strategy="join").alias("top_3_fastest")
])
```

---

## 6. SORT CONTEXT

**Behavior**: Orders rows by one or more columns.

### Basic Sorting

```python
# Single column
df.sort("date")
df.sort("revenue", descending=True)

# Multiple columns
df.sort(["country", "state", "city"])
df.sort(["department", "salary"], descending=[False, True])

# Nulls placement
df.sort("value", nulls_last=True)
```

### Sorting by Expression

```python
# Sort by computed expression
df.sort(pl.col("revenue") / pl.col("cost"))

# Sort with multiple expressions
df.sort([
    pl.col("priority").cast(pl.Int32),
    pl.col("created_at")
])
```

### Sort Interactions with Other Contexts

```python
# Sort before group_by to control first/last in aggregation
df.sort("date", descending=True).group_by("customer_id").agg([
    pl.col("order_id").first().alias("latest_order")
])

# Sort within aggregation (doesn't affect outer order)
df.group_by("category").agg([
    pl.col("product").sort_by("sales", descending=True).first()
])
```

---

## 7. BASIC OPERATIONS IN EXPRESSIONS

### Arithmetic Operations

```python
# Basic arithmetic
pl.col("a") + pl.col("b")
pl.col("a") - 5
pl.col("price") * 1.1
pl.col("total") / pl.col("count")
pl.col("x") ** 2  # Power
pl.col("x") % 3   # Modulo

# Named versions (Python)
pl.col("a").add(pl.col("b"))
pl.col("a").sub(5)
pl.col("price").mul(1.1)
pl.col("total").truediv(pl.col("count"))
pl.col("x").pow(2)
pl.col("x").mod(3)
```

### Comparison Operations

```python
pl.col("x") > 5
pl.col("x") >= 5
pl.col("x") < 5
pl.col("x") <= 5
pl.col("x") == 5
pl.col("x") != 5

# Named versions
pl.col("x").gt(5)
pl.col("x").ge(5)
pl.col("x").lt(5)
pl.col("x").le(5)
pl.col("x").eq(5)
pl.col("x").ne(5)
```

### Boolean and Bitwise Operations

```python
# Boolean (use &, |, ~ not and, or, not)
(pl.col("x") > 0) & (pl.col("y") < 10)
(pl.col("status") == "A") | (pl.col("status") == "B")
~pl.col("is_valid")

# Named boolean functions
pl.col("x").gt(0).and_(pl.col("y").lt(10))
pl.col("status").eq("A").or_(pl.col("status").eq("B"))
pl.col("is_valid").not_()

# Bitwise operations
pl.col("flags") & 0xFF
pl.col("flags") | 0x01
~pl.col("bits")
pl.col("a") ^ pl.col("b")  # XOR

# Named bitwise functions
pl.col("flags").and_(0xFF)
pl.col("flags").or_(0x01)
pl.col("bits").not_()
pl.col("a").xor(pl.col("b"))
```

### Conditional Expressions

```python
# Basic when-then-otherwise
pl.when(pl.col("age") >= 18)
  .then("adult")
  .otherwise("minor")

# Multiple conditions (if-elif-else chain)
pl.when(pl.col("score") >= 90)
  .then("A")
  .when(pl.col("score") >= 80)
  .then("B")
  .when(pl.col("score") >= 70)
  .then("C")
  .otherwise("F")

# Nested conditions
pl.when(pl.col("type") == "premium")
  .then(
      pl.when(pl.col("tenure") > 5)
        .then(0.20)
        .otherwise(0.15)
  )
  .otherwise(0.10)

# Using expressions in conditions
pl.when(pl.col("value").is_null())
  .then(pl.col("default_value"))
  .otherwise(pl.col("value"))
```

### Counting and Unique Values

```python
# Count operations
pl.len()  # Count all rows (including nulls)
pl.col("id").count()  # Count non-null values
pl.col("id").null_count()  # Count nulls
pl.col("category").n_unique()  # Count distinct values
pl.col("category").approx_n_unique()  # Fast approximate count

# Value counts (returns struct)
pl.col("status").value_counts()
# Result: struct with "status" and "count" fields

# Get unique values
pl.col("category").unique()
pl.col("category").unique(maintain_order=True)

# Get counts of unique values (corresponding to .unique() order)
pl.col("category").unique_counts()
```

---

## 8. CASTING AND TYPE CONVERSIONS

### Basic Casting

```python
# Numeric conversions
pl.col("integers").cast(pl.Float64)
pl.col("floats").cast(pl.Int32)  # Truncates decimals

# String conversions
pl.col("numbers").cast(pl.String)
pl.col("string_numbers").cast(pl.Int32)

# Boolean conversions
pl.col("integers").cast(pl.Boolean)  # 0 -> False, others -> True
pl.col("booleans").cast(pl.Int32)    # False -> 0, True -> 1
```

### Downcasting (Memory Optimization)

```python
# Reduce precision to save memory
pl.col("id").cast(pl.Int32)      # i64 -> i32
pl.col("price").cast(pl.Float32)  # f64 -> f32

# Check memory usage
print(f"Before: {df.estimated_size()} bytes")
df_optimized = df.with_columns([
    pl.col("small_int").cast(pl.Int16),
    pl.col("price").cast(pl.Float32)
])
print(f"After: {df_optimized.estimated_size()} bytes")
```

### Strict vs Non-Strict Casting

```python
# strict=True (DEFAULT): Raises error on conversion failure
try:
    df.select(pl.col("mixed").cast(pl.Int32, strict=True))
except pl.exceptions.InvalidOperationError as e:
    print(f"Conversion failed: {e}")

# strict=False: Converts failures to null
df.select(pl.col("mixed").cast(pl.Int32, strict=False))
# Values that can't convert become null
```

### Temporal Type Conversions

```python
# Date/Datetime to integers (days/microseconds since epoch)
pl.col("date").cast(pl.Int64)      # Days since 1970-01-01
pl.col("datetime").cast(pl.Int64)  # Microseconds since epoch
pl.col("time").cast(pl.Int64)      # Nanoseconds since midnight

# Integers to temporal types
pl.col("days_since_epoch").cast(pl.Date)
pl.col("us_since_epoch").cast(pl.Datetime("us"))

# String to temporal (with format)
pl.col("date_string").str.to_date("%Y-%m-%d")
pl.col("datetime_string").str.to_datetime("%Y-%m-%d %H:%M:%S")

# Temporal to string (with format)
pl.col("date").dt.to_string("%Y-%m-%d")
pl.col("datetime").dt.to_string("%Y-%m-%d %H:%M:%S")
```

---

## 9. COMBINING CONTEXTS

### Typical Query Pattern

```python
result = (
    df.lazy()

    # 1. Filter: Reduce data early
    .filter(
        (pl.col("year") == 2024) &
        (pl.col("status") == "completed")
    )

    # 2. With_columns: Add derived columns
    .with_columns([
        (pl.col("revenue") - pl.col("cost")).alias("profit"),
        pl.col("date").dt.month().alias("month")
    ])

    # 3. Group_by + Agg: Aggregate by groups
    .group_by(["region", "month"])
    .agg([
        pl.col("profit").sum().alias("total_profit"),
        pl.col("profit").mean().alias("avg_profit"),
        pl.col("customer_id").n_unique().alias("unique_customers")
    ])

    # 4. With_columns: Post-aggregation calculations
    .with_columns([
        (pl.col("total_profit") / pl.col("unique_customers")).alias("profit_per_customer")
    ])

    # 5. Sort: Order results
    .sort("total_profit", descending=True)

    # 6. Select: Choose final output columns
    .select([
        "region",
        "month",
        "total_profit",
        "profit_per_customer"
    ])

    .collect()
)
```

### Performance Tips

```python
# GOOD: Filter early to reduce data processed
df.filter(pl.col("year") == 2024).group_by("category").agg(...)

# BAD: Filter after expensive operations
df.group_by("category").agg(...).filter(pl.col("year") == 2024)

# GOOD: Use lazy evaluation with .lazy() and .collect()
result = df.lazy().filter(...).group_by(...).agg(...).collect()

# GOOD: Generate all expressions at once
exprs = [pl.col(c) * 2 for c in columns]
df.with_columns(exprs)

# BAD: Multiple separate context calls
for c in columns:
    df = df.with_columns(pl.col(c) * 2)  # Inefficient
```

---

## 10. COMMON PATTERNS AND IDIOMS

### Working with Nulls

```python
# Fill nulls
pl.col("value").fill_null(0)
pl.col("value").fill_null(strategy="forward")
pl.col("value").fill_null(strategy="backward")
pl.col("value").fill_null(pl.col("default_value"))

# Drop nulls
pl.col("value").drop_nulls()

# Check for nulls
pl.col("value").is_null()
pl.col("value").is_not_null()

# Null handling in aggregations
pl.col("value").drop_nulls().mean()  # Exclude nulls from mean
```

### String Operations

```python
# Common string methods
pl.col("text").str.to_uppercase()
pl.col("text").str.to_lowercase()
pl.col("text").str.strip_chars()
pl.col("text").str.replace("old", "new")
pl.col("text").str.replace_all("pattern", "replacement")
pl.col("text").str.slice(0, 10)
pl.col("text").str.lengths()

# String predicates
pl.col("text").str.contains("pattern")
pl.col("text").str.starts_with("prefix")
pl.col("text").str.ends_with("suffix")

# String splitting
pl.col("text").str.split(",")
pl.col("text").str.split(",").list.get(0)  # First element
```

### Working with Lists

```python
# List operations
pl.col("list_col").list.lengths()
pl.col("list_col").list.get(0)  # First element
pl.col("list_col").list.first()
pl.col("list_col").list.last()
pl.col("list_col").list.sum()
pl.col("list_col").list.mean()
pl.col("list_col").list.contains(5)

# Explode list into rows
df.explode("list_col")

# Create list from multiple columns
pl.concat_list(["col1", "col2", "col3"]).alias("combined_list")
```

### Date/Time Operations

```python
# Extract components
pl.col("date").dt.year()
pl.col("date").dt.month()
pl.col("date").dt.day()
pl.col("datetime").dt.hour()
pl.col("datetime").dt.minute()
pl.col("datetime").dt.weekday()

# Date arithmetic
pl.col("date") + pl.duration(days=7)
pl.col("date") - pl.duration(weeks=1)

# Truncate/round
pl.col("datetime").dt.truncate("1h")
pl.col("datetime").dt.round("1d")

# Between dates
pl.col("date").is_between(start_date, end_date)
```

### Type Checking in Expressions

```python
# Check column types
df.select([
    pl.col(pl.Utf8),      # Select string columns
    pl.col(pl.NUMERIC_DTYPES),  # Select numeric columns
])

# Conditional based on type (in your logic, not Polars expressions)
numeric_cols = [col for col in df.columns if df[col].dtype in pl.NUMERIC_DTYPES]
df.select([pl.col(c).sum() for c in numeric_cols])
```

---

## 11. ANTI-PATTERNS TO AVOID

### âŒ Using Python Loops Instead of Expression Expansion

```python
# BAD
result = df
for col in ["a", "b", "c"]:
    result = result.with_columns(pl.col(col) * 2)

# GOOD
df.with_columns([pl.col(col) * 2 for col in ["a", "b", "c"]])
df.with_columns(pl.col("a", "b", "c") * 2)
```

### âŒ Using Python Functions Instead of Expressions

```python
# BAD (slow, loses parallelization)
df.select(pl.col("value").map_elements(lambda x: x * 2))

# GOOD
df.select(pl.col("value") * 2)
```

### âŒ Not Using Lazy Evaluation

```python
# BAD (eager evaluation, less optimization)
df.filter(...).group_by(...).agg(...).sort(...)

# GOOD (lazy evaluation, optimized by query planner)
df.lazy().filter(...).group_by(...).agg(...).sort(...).collect()
```

### âŒ Filtering After Expensive Operations

```python
# BAD
df.group_by("id").agg(expensive_calculation()).filter(pl.col("year") == 2024)

# GOOD
df.filter(pl.col("year") == 2024).group_by("id").agg(expensive_calculation())
```

### âŒ Creating Duplicate Column Names

```python
# BAD (causes error in select context)
df.select([
    pl.col("price"),
    pl.col("price") * 1.1  # Still named "price" - error!
])

# GOOD
df.select([
    pl.col("price"),
    (pl.col("price") * 1.1).alias("price_with_tax")
])
```

### âŒ Confusion Between .over() and .group_by()

```python
# UNDERSTAND THE DIFFERENCE

# over(): Keeps all rows, broadcasts aggregation
df.select(pl.col("value").mean().over("group"))
# Result has same rows as input

# group_by(): One row per group
df.group_by("group").agg(pl.col("value").mean())
# Result has n_unique("group") rows
```

---

## 12. EXPRESSION OPTIMIZATION TIPS

### Predicate Pushdown

```python
# Polars automatically pushes filters down in lazy evaluation
df.lazy().filter(pl.col("year") == 2024).select(...).collect()
# Filter applied as early as possible, even if select comes first in code
```

### Projection Pushdown

```python
# Only required columns are read/computed
df.lazy().select(["id", "name"]).filter(...).collect()
# Only "id" and "name" columns processed
```

### Common Subexpression Elimination

```python
# Polars detects and reuses common expressions
df.lazy().with_columns([
    (pl.col("a") + pl.col("b")).alias("c"),
    (pl.col("a") + pl.col("b") + pl.col("d")).alias("e")
]).collect()
# "pl.col('a') + pl.col('b')" computed once, reused
```

### Parallel Execution

```python
# Group_by aggregations automatically parallelized
df.group_by("category").agg([
    pl.col("value").sum(),
    pl.col("value").mean(),
    pl.col("value").std()
])
# Aggregations computed in parallel across groups
```

---

## 13. DEBUGGING EXPRESSIONS

### Inspect Lazy Query Plan

```python
lf = df.lazy().filter(...).group_by(...).agg(...)

# See unoptimized plan
print(lf.show_graph())

# See optimized plan
print(lf.show_graph(optimized=True))

# Explain plan as text
print(lf.explain())
```

### Schema Validation

```python
# Check output schema before collecting
lf = df.lazy().select([...])
print(lf.collect_schema())

# Validate expected schema
expected_schema = {
    "name": pl.Utf8,
    "age": pl.Int32,
    "score": pl.Float64
}
assert lf.collect_schema() == expected_schema
```

### Expression Testing

```python
# Test expression on sample data
sample = df.head(100)
result = sample.select(your_complex_expression)
print(result)

# Use describe() to understand data
df.select(pl.col("value")).describe()
```

---

## 14. QUICK REFERENCE TABLE

| Task            | Expression Pattern                      |
| --------------- | --------------------------------------- |
| Select columns  | `pl.col("name")`, `pl.col("a", "b")`    |
| All columns     | `pl.all()`                              |
| By dtype        | `pl.col(pl.Float64)`                    |
| By pattern      | `pl.col("^sales_.*$")`                  |
| Exclude         | `pl.all().exclude("id")`                |
| Alias           | `.alias("new_name")`                    |
| Cast type       | `.cast(pl.Float64)`                     |
| Fill null       | `.fill_null(0)`                         |
| Drop null       | `.drop_nulls()`                         |
| Conditional     | `pl.when(...).then(...).otherwise(...)` |
| Filter          | `(condition1) & (condition2)`           |
| Sum             | `.sum()`                                |
| Mean            | `.mean()`                               |
| Count           | `pl.len()`, `.count()`                  |
| Unique count    | `.n_unique()`                           |
| Group by        | `df.group_by("col").agg([...])`         |
| Window          | `.over("group")`                        |
| Sort            | `df.sort("col", descending=True)`       |
| String upper    | `.str.to_uppercase()`                   |
| String contains | `.str.contains("pattern")`              |
| Date year       | `.dt.year()`                            |
| List length     | `.list.lengths()`                       |
| Rank            | `.rank()`                               |
| Cumulative sum  | `.cum_sum()`                            |

---

## SUMMARY

**Key Takeaways:**

1. **Context determines behavior**: The same expression produces different results in different contexts
2. **select() returns only specified columns**, with_columns() keeps all columns
3. **filter() expects boolean expressions**, group_by() expects grouping keys
4. **over() maintains row count**, group_by() reduces to one row per group
5. **Use lazy evaluation** with .lazy() and .collect() for optimization
6. **Avoid Python loops** - generate all expressions at once
7. **Stay in expression API** - avoid .map_elements() when possible
8. **Expression expansion** is powerful - use pl.col() with types, patterns, and selectors
9. **Null handling** is explicit - nulls in comparisons evaluate to null (falsy)
10. **Performance matters** - filter early, use appropriate contexts, leverage parallelization

This guide covers the essential patterns for working with Polars expression contexts effectively.
