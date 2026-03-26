from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.score_parsers import TECHNIQUE_ORDER, parse_client_scores_excel


def test_parse_client_scores_excel_with_class_value(tmp_path: Path) -> None:
    excel_path = tmp_path / "sample.xlsx"
    frame = pd.DataFrame(
        {
            "Class": TECHNIQUE_ORDER,
            "Value": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        }
    )
    frame.to_excel(excel_path, index=False)

    scores = parse_client_scores_excel(excel_path)
    assert scores == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]


def test_parse_client_scores_excel_with_skill_tag_score(tmp_path: Path) -> None:
    excel_path = tmp_path / "sample.xlsx"
    frame = pd.DataFrame(
        {
            "Skill Tag": TECHNIQUE_ORDER,
            "Score": [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
        }
    )
    frame.to_excel(excel_path, index=False)

    scores = parse_client_scores_excel(excel_path)
    assert scores == [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
