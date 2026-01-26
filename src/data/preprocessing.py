import polars as pl

def drop_unused_columns(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {
        "job_market": ["publication_date"]
    }
    return df.drop(config.get(dataset, []))

def _remove_outliers_iqr(df: pl.DataFrame, column: str, threshold: float = 1.5) -> pl.DataFrame:
    return df.filter(
        pl.col(column).is_between(
            pl.col(column).quantile(0.25) - threshold * (pl.col(column).quantile(0.75) - pl.col(column).quantile(0.25)),
            pl.col(column).quantile(0.75) + threshold * (pl.col(column).quantile(0.75) - pl.col(column).quantile(0.25))
        )
    )

def clean_outliers(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {
        "job_market": ["salary_max", "salary_min"]
    }

    result = df

    for col in config.get(dataset, []):
        result = _remove_outliers_iqr(result, col)

    return result

def handle_nulls(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {
        "job_market": {
            "categorical": ["job_type", "category"],
            "numeric": ["experience_required"]
        }
    }

    cfg = config.get(dataset)

    if cfg is None:
        return df

    exprs = []

    for col in cfg.get("categorical", []):
        exprs.append(
            pl.col(col).fill_null("Unknown")
        )

    for col in cfg.get("numeric", []):
        exprs.append(
            pl.col(col).fill_null(pl.col(col).median())
        )

    return df.with_columns(exprs)

def add_salary_mean(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        ((pl.col("salary_min") + pl.col("salary_max")) / 2).alias("salary_mean")
    ]).drop(["salary_min", "salary_max"])

def one_hot_encode(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {
        "job_market": ["job_type", "category"]
    }
    
    result = df

    for col in config.get(dataset, []):

        dummies = result.select(pl.col(col)).to_dummies()
        dummy_cols = dummies.columns
        if len(dummy_cols) > 1:
            dummies = dummies.select(dummy_cols[1:])

        result = result.with_columns(dummies).drop(col)

    return result

def mean_target_encode(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {
        "job_market": [["salary_mean"], ["job_title", "company", "location"]]
    }
    
    target_col = config.get(dataset, [])[0]
    result = df

    for col in config.get(dataset, [])[1]:

        mean_table = (
            result.group_by(col)
              .agg(pl.col(target_col).mean().alias(f"{col}_mte"))
        )

        result = (
            result.join(mean_table, on=col, how="left")
              .drop(col)
        )

    return result

def preprocess_skills(df: pl.DataFrame) -> pl.DataFrame:
    column = "skills"
    sep = ","
    tmp = "_skills_list"

    df_skills = df.with_columns(
        pl.col(column)
        .fill_null("")
        .str.strip_chars('" ')
        .str.split(sep)
        .alias(tmp)
    )

    all_skills = (
        df_skills
        .select(pl.col(tmp).explode())
        .with_columns(pl.col(tmp).str.strip_chars())
        .filter(pl.col(tmp) != "") 
        .unique()
        .to_series()
        .to_list()
    )

    for skill in all_skills:
        df_skills = df_skills.with_columns(
            pl.col(tmp)
            .list.eval(pl.element().str.strip_chars())
            .list.contains(skill)
            .cast(pl.Int8)
            .alias(skill)
        )

    return df_skills.drop(tmp).drop(column)

def preprocess(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    result = df

    match dataset:
        case "job_market":
            result = drop_unused_columns(result, dataset)
            result = clean_outliers(result, dataset)
            result = handle_nulls(result, dataset)
            result = add_salary_mean(result)
            result = one_hot_encode(result, dataset)
            result = mean_target_encode(result, dataset)
            result = preprocess_skills(result)
        case _:
            pass

    return result