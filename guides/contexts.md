# Expression Contexts

Expressions behave differently depending on their context.

## 1. Select Context

Transform or compute columns. Returns specified columns only.

```python
lf.select([
    pl.col("name"),
    pl.col("salary") * 1.1,
    (pl.col("bonus") + pl.col("commission")).alias("extra")
])
```

## 2. With Columns Context

Add or modify columns. Keeps all existing columns.

```python
lf.with_columns([
    pl.col("price").cast(pl.Float64),
    (pl.col("quantity") * pl.col("price")).alias("total")
])
```

## 3. Filter Context

Boolean expressions for row filtering.

```python
lf.filter(
    (pl.col("age") >= 18) &
    (pl.col("country") == "US")
)
```

Multiple conditions:

```python
lf.filter(
    pl.col("status").is_in(["active", "pending"]) &
    pl.col("amount") > 1000
)
```

## 4. Group By + Agg Context

Aggregations over groups.

```python
lf.group_by("department").agg([
    pl.col("salary").mean().alias("avg_salary"),
    pl.col("employee_id").count().alias("headcount"),
    pl.col("salary").max().alias("max_salary")
])
```

Maintain order:

```python
lf.group_by("category", maintain_order=True).agg(...)
```

## 5. Sort Context

```python
lf.sort("date", descending=True)
lf.sort(["department", "salary"], descending=[False, True])
```

## Combining Contexts

```python
result = (
    lf
    .filter(pl.col("year") == 2024)           # Filter
    .with_columns([                           # Add columns
        (pl.col("revenue") - pl.col("cost")).alias("profit")
    ])
    .group_by("region")                       # Group
    .agg([                                    # Aggregate
        pl.col("profit").sum(),
        pl.col("profit").mean()
    ])
    .sort("profit_sum", descending=True)      # Sort
    .select([                                 # Select final
        pl.col("region"),
        pl.col("profit_sum")
    ])
    .collect()
)
```

## Expression Expansion

Some expressions expand to multiple columns:

```python
# Select all numeric columns
lf.select(pl.col(pl.NUMERIC_DTYPES))

# Apply to all except one
lf.select(pl.all().exclude("id"))

# Multiple transformations
lf.select(pl.col("^sales_.*$") * 1.1)
```
