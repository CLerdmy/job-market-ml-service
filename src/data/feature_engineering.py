import numpy as np
import polars as pl


def _get_skills_groups() -> dict:
    return {
        "backend_skills": ["Python", "Java", "Go", "Ruby", "Node.js"],
        "frontend_skills": ["JavaScript", "TypeScript", "React"],
        "db_skills": ["SQL", "MongoDB"],
        "ml_skills": ["Machine Learning", "TensorFlow"],
        "infra_skills": ["AWS", "Docker", "Kubernetes", "CI/CD"],
        "tools_skills": ["Git", "Agile", "REST APIs"]
    }

def aggregate_skills(df: pl.DataFrame) -> pl.DataFrame:
    skill_groups = _get_skills_groups()

    all_skills = [skill for skills in skill_groups.values() for skill in skills]
    missing_skills = [c for c in all_skills if c not in df.columns]
    if missing_skills:
        df = df.with_columns([pl.lit(0).cast(pl.Int8).alias(c) for c in missing_skills])

    df = df.with_columns([
        pl.sum_horizontal([pl.col(skill) for skill in skills]).alias(group)
        for group, skills in skill_groups.items()
    ])

    return df

def add_skill_count(df: pl.DataFrame) -> pl.DataFrame:
    skill_groups = _get_skills_groups()

    all_skills = [skill for skills in skill_groups.values() for skill in skills]

    df = df.with_columns([
        pl.sum_horizontal([pl.col(c) for c in all_skills]).alias("skill_count")
    ])

    return df

def add_experience_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("experience_required") ** 2).alias("experience_sq"),
        pl.col("experience_required")
          .log1p()
          .alias("experience_log")
    ])

def drop_all_skill_cols(df: pl.DataFrame) -> pl.DataFrame:
    skill_groups = _get_skills_groups()

    all_skill_cols = [skill for skills in skill_groups.values() for skill in skills]
    df = df.drop(all_skill_cols)

    return df

def drop_experience_required_col(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop("experience_required")

def build_features(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    result = df

    match dataset:
        case "job_market":
            result = aggregate_skills(result)
            result = add_skill_count(result)
            result = drop_all_skill_cols(result)
            result = add_experience_features(result)
            result = drop_experience_required_col(result)
        case _:
            pass

    return result