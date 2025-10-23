# Polars Expression System

## Core Concept

Expressions are **lazy operations** that compose together. They map Series → Series.

## Composition Methods

### 1. Method Chaining

```python
pl.col("value").cast(pl.Int32).abs().sum()
```

### 2. Operators

```python
pl.col("a") + pl.col("b")
pl.col("price") * 1.1
pl.col("quantity") / pl.col("total")
```

### 3. Nesting

```python
# Filter within aggregation
pl.col("sales").filter(pl.col("region") == "US").sum()

# Conditional in expression
pl.when(pl.col("age") > 18).then(pl.col("income")).sum()
```

## Column Selection

```python
pl.col("name")           # Single column
pl.col(["a", "b"])       # Multiple columns
pl.col("^sales_.*$")     # Regex pattern
pl.all()                 # All columns
pl.exclude("id")         # All except
```

## Common Patterns

### Conditional Logic

```python
pl.when(pl.col("score") > 90)
  .then(pl.lit("A"))
  .when(pl.col("score") > 80)
  .then(pl.lit("B"))
  .otherwise(pl.lit("C"))
```

### Aggregations

```python
pl.col("value").sum()
pl.col("value").mean()
pl.col("value").min()
pl.col("value").max()
pl.col("value").count()
```

### Window Functions

```python
# Running sum
pl.col("value").cum_sum()

# Over partition
pl.col("sales").sum().over("region")

# Rolling window
pl.col("price").rolling_mean(window_size=7)
```

### String Operations

```python
pl.col("name").str.to_lowercase()
pl.col("text").str.contains("pattern")
pl.col("url").str.extract(r"https://(.+)", 1)
```

## Multiple Expressions in Parallel

```python
lf.select([
    pl.col("a").sum(),
    pl.col("b").mean(),
    pl.col("c").max()
])
# All three run in parallel
```

## Expression Aliases

```python
pl.col("value").sum().alias("total_value")
pl.col("a").cast(pl.Float64).alias("a_float")
```
