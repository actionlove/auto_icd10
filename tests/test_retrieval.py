import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from icd10_pipeline.retrieval import ICD10Index

TABLE = Path(__file__).parent.parent / "data" / "icd10cm_codes_sample.csv"


def test_loads_table():
    idx = ICD10Index(TABLE)
    assert len(idx) >= 10


def test_finds_hypertension():
    idx = ICD10Index(TABLE)
    codes = [c for c, _, _ in idx.search("essential hypertension", k=5)]
    assert "I10" in codes


def test_finds_diabetes_hyperglycemia():
    idx = ICD10Index(TABLE)
    codes = [c for c, _, _ in idx.search("type 2 diabetes with hyperglycemia", k=3)]
    assert "E11.65" in codes


def test_empty_query():
    idx = ICD10Index(TABLE)
    assert idx.search("") == []
