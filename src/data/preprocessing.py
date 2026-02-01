from typing import Any, Dict, Optional

import polars as pl


def drop_unused_columns(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {"job_market": ["publication_date"]}
    return df.drop(config.get(dataset, []))


def _remove_outliers_iqr(
    df: pl.DataFrame, column: str, threshold: float = 1.5
) -> pl.DataFrame:
    return df.filter(
        pl.col(column).is_between(
            pl.col(column).quantile(0.25)
            - threshold
            * (pl.col(column).quantile(0.75) - pl.col(column).quantile(0.25)),
            pl.col(column).quantile(0.75)
            + threshold
            * (pl.col(column).quantile(0.75) - pl.col(column).quantile(0.25)),
        )
    )


def clean_outliers(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {"job_market": ["salary_max", "salary_min"]}

    result = df

    for col in config.get(dataset, []):
        result = _remove_outliers_iqr(result, col)

    return result


def handle_nulls(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {
        "job_market": {
            "categorical": ["job_type", "category"],
            "numeric": ["experience_required"],
        }
    }

    cfg = config.get(dataset)

    if cfg is None:
        return df

    exprs = []

    for col in cfg.get("categorical", []):
        exprs.append(pl.col(col).fill_null("Unknown"))

    for col in cfg.get("numeric", []):
        exprs.append(pl.col(col).fill_null(pl.col(col).median()))

    return df.with_columns(exprs)


def add_salary_mean(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [((pl.col("salary_min") + pl.col("salary_max")) / 2).alias("salary_mean")]
    ).drop(["salary_min", "salary_max"])


def one_hot_encode(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {"job_market": ["job_type", "category"]}

    result = df

    for col in config.get(dataset, []):
        dummies = result.select(pl.col(col)).to_dummies()
        dummy_cols = dummies.columns
        if len(dummy_cols) > 1:
            dummies = dummies.select(dummy_cols[1:])

        result = result.with_columns(dummies).drop(col)

    return result


def mean_target_encode(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    config = {"job_market": [["salary_mean"], ["job_title", "company", "location"]]}

    target_col = config.get(dataset, [])[0]
    result = df

    for col in config.get(dataset, [])[1]:
        mean_table = result.group_by(col).agg(
            pl.col(target_col).mean().alias(f"{col}_mte")
        )

        result = result.join(mean_table, on=col, how="left").drop(col)

    return result


def preprocess_skills(df: pl.DataFrame) -> pl.DataFrame:
    column = "skills"
    sep = ","
    tmp = "_skills_list"

    df_skills = df.with_columns(
        pl.col(column).fill_null("").str.strip_chars('" ').str.split(sep).alias(tmp)
    )

    all_skills = (
        df_skills.select(pl.col(tmp).explode())
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


def apply_mte(df: pl.DataFrame, mte: Dict[str, Dict[Any, float]]) -> pl.DataFrame:
    result = df
    for col, mapping in mte.items():
        result = result.with_columns(
            pl.col(col).replace(mapping, default=None).alias(f"{col}_mte")
        )
    return result


def use_salary_predictor_columns(df: pl.DataFrame) -> pl.DataFrame:
    model_features = [
        "job_type_Full_time",
        "job_type_Full-time",
        "job_type_Internship",
        "job_type_Part-time",
        "job_type_Remote",
        "job_type_Unknown",
        "job_type_Working_student",
        "job_type_berufseinstieg",
        "job_type_berufserfahren",
        "job_type_manager",
        "job_type_professional_/_experienced",
        "category_HR",
        "category_Helpdesk",
        "category_Marketing_and_Communication",
        "category_Media_Planning",
        "category_Process_Engineering",
        "category_Recruitment_and_Selection",
        "category_Remote",
        "category_SAP/ERP_Consulting",
        "category_Social_Media_Manager",
        "category_Software_Development",
        "category_Technology",
        "category_Unknown",
        "job_title_mte",
        "company_mte",
        "location_mte",
        "backend_skills",
        "frontend_skills",
        "db_skills",
        "ml_skills",
        "infra_skills",
        "tools_skills",
        "skill_count",
        "experience_sq",
        "experience_log",
    ]

    result = df

    result = result.select([c for c in model_features if c in result.columns])

    missing_cols = [c for c in model_features if c not in result.columns]
    if missing_cols:
        zeros_df = pl.DataFrame({c: [0] * result.height for c in missing_cols})
        result = result.hstack(zeros_df)

    result = result.select(model_features)

    return result


def preprocess(
    df: pl.DataFrame,
    dataset: str,
    mte: Optional[Dict[str, Dict[Any, float]]] = None,
    train: bool = True,
) -> pl.DataFrame:
    result = df

    if not train:
        column_mapping = {"experience": "experience_required", "work_type": "job_type"}
        existing_columns = [c for c in column_mapping if c in result.columns]
        if existing_columns:
            result = result.rename({c: column_mapping[c] for c in existing_columns})

    match dataset:
        case "job_market":
            if train:
                result = drop_unused_columns(result, dataset)
                result = clean_outliers(result, dataset)
                result = handle_nulls(result, dataset)
                result = add_salary_mean(result)
                result = one_hot_encode(result, dataset)
                if mte is None:
                    result = mean_target_encode(result, dataset)
                else:
                    result = apply_mte(result, mte)
                result = preprocess_skills(result)

            elif not train:
                result = handle_nulls(result, dataset)
                result = one_hot_encode(result, dataset)
                if mte is not None:
                    result = apply_mte(result, mte)
                result = preprocess_skills(result)

        case _:
            pass

    return result
