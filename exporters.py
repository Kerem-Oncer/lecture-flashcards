"""Export reviewed flashcards as CSV or an Anki deck."""

import hashlib
import html
import re
import tempfile
from collections import Counter
from pathlib import Path

import genanki
import pandas as pd

EXPORT_COLUMNS = ["front", "back", "card_type", "source_quote", "tags"]

# Fixed so that re-importing reuses the same note type in Anki. Don't change it.
MODEL_ID = 1738294015

ANKI_MODEL = genanki.Model(
    MODEL_ID,
    "Lecture Flashcards",
    fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Source"}],
    templates=[
        {
            "name": "Card",
            "qfmt": "{{Front}}",
            "afmt": (
                '{{FrontSide}}<hr id="answer">{{Back}}'
                '{{#Source}}<div class="source">{{Source}}</div>{{/Source}}'
            ),
        }
    ],
    css=(
        ".card { font-family: arial; font-size: 20px; text-align: center; }\n"
        ".source { font-size: 13px; color: #888888; margin-top: 16px; }"
    ),
)


def _clean_text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def clean_cards(cards_df) -> pd.DataFrame:
    """Return the export columns, without rows missing a front or back."""
    cards = pd.DataFrame(cards_df, columns=EXPORT_COLUMNS).copy()
    for column in ["front", "back", "card_type", "source_quote"]:
        cards[column] = cards[column].map(_clean_text)
    return cards[(cards["front"] != "") & (cards["back"] != "")].reset_index(drop=True)


def _anki_tag(tag) -> str:
    return re.sub(r"\s+", "_", str(tag).strip())


def _tag_list(tags) -> list[str]:
    """Normalize a tags cell, which can be a list, an array, a string, or empty."""
    if isinstance(tags, str):
        tags = [tags]
    try:
        return [_anki_tag(tag) for tag in tags if str(tag).strip()]
    except TypeError:  # None or NaN, from a row added by hand.
        return []


def _to_field(text: str) -> str:
    """Escape text for Anki, whose fields are HTML, so <stdio.h> isn't eaten as a tag."""
    return html.escape(text, quote=False).replace("\n", "<br>")


def _deck_id(deck_name: str) -> int:
    # hashlib, because Python's hash() gives different results in each process.
    digest = int(hashlib.sha256(deck_name.encode("utf-8")).hexdigest(), 16)
    return (1 << 30) + digest % (1 << 30)


def to_csv(cards_df) -> bytes:
    """Return the cards as UTF-8 CSV, with space-separated tags as Anki expects."""
    cards = clean_cards(cards_df)
    cards["tags"] = cards["tags"].map(lambda tags: " ".join(_tag_list(tags)))
    return cards.to_csv(index=False).encode("utf-8")


def to_apkg(cards_df, deck_name: str) -> bytes:
    """Return the cards as an Anki .apkg file."""
    deck = genanki.Deck(_deck_id(deck_name), deck_name)
    occurrences = Counter()
    for card in clean_cards(cards_df).to_dict("records"):
        # Anki matches notes by GUID on re-import. Basing it on the front means an
        # edited back updates the note, and the counter keeps identical fronts apart.
        occurrences[card["front"]] += 1
        tags = [_anki_tag(deck_name), _anki_tag(card["card_type"]), *_tag_list(card["tags"])]
        deck.add_note(
            genanki.Note(
                model=ANKI_MODEL,
                fields=[
                    _to_field(card["front"]),
                    _to_field(card["back"]),
                    _to_field(card["source_quote"]),
                ],
                tags=list(dict.fromkeys(tag for tag in tags if tag)),
                guid=genanki.guid_for(deck_name, card["front"], occurrences[card["front"]]),
            )
        )

    # genanki can only write to a path. A temp directory also avoids Windows
    # file-locking problems that NamedTemporaryFile can hit.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "deck.apkg"
        genanki.Package(deck).write_to_file(str(path))
        return path.read_bytes()
