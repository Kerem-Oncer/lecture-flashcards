"""Tests for the study view's card JSON, which carries untrusted text into a <script> tag."""

import json

import pandas as pd

from study import build_cards_json, build_study_html

ATTACK = "<script>alert(1)</script>"


def cards_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"front": "What is TCP?", "back": "A reliable protocol.", "source_quote": "TCP is reliable"},
            {"front": "", "back": "No front, so skipped.", "source_quote": ""},
            {"front": "No back, so skipped.", "back": None, "source_quote": ""},
            {"front": ATTACK, "back": "Tom & Jerry <b>bold</b>", "source_quote": "</script><!--"},
        ]
    )


def test_cards_json_skips_rows_missing_front_or_back():
    cards = json.loads(build_cards_json(cards_df()))

    assert [card["front"] for card in cards] == ["What is TCP?", ATTACK]
    # The escapes decode back to the original text, which the page sets via textContent.
    assert cards[1] == {
        "front": ATTACK,
        "back": "Tom & Jerry <b>bold</b>",
        "source_quote": "</script><!--",
    }


def test_script_text_in_a_card_is_never_unescaped_in_the_html():
    page = build_study_html(cards_df())

    assert ATTACK not in page
    assert "<b>bold</b>" not in page
    assert "<!--" not in page
    assert page.count("</script>") == 1  # Only the template's own closing tag.
    assert "innerHTML" not in page
    assert "textContent" in page


def test_theme_is_set_on_the_page():
    assert 'data-theme="dark"' in build_study_html(cards_df(), dark=True)
    assert 'data-theme="light"' in build_study_html(cards_df(), dark=False)
