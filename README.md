# Lecture → Flashcards

Turn messy lecture notes, PDFs, and YouTube transcripts into source-grounded flashcards you can review, edit, and export to Anki.

**Live demo:** [Open the app](https://lecture-flashcards-ckr8hemndpmqnbx5tvz2eo.streamlit.app/)

![Card preview showing questions, answers, and source quotes](docs/screenshot.png)

**Built with:** Python, Streamlit, Claude API, pydantic, pypdf, genanki, pandas

## How it works

Extract text (pypdf) → LLM generates JSON (Claude API) → validated with pydantic, with one retry on error → human review in an editable table → export to Anki `.apkg` or CSV

## Design decisions

- **Source quotes for grounding:** the model may only use the provided notes, and each card includes a short quote from them, so answers can be checked against the source during review.
- **Schema validation with retry:** every response is validated against a pydantic schema. On failure, the exact error goes back to the model for one retry, and a second failure shows a clear error.
- **Human review before export:** cards land in an editable table, so nothing reaches a study deck without a chance to fix or delete it.
- **Stable genanki IDs:** a fixed note type ID, a deck ID hashed from the deck name, and note GUIDs based on the card front mean re-importing updates cards instead of duplicating them.
- **Chunking for long notes:** text is split into chunks of about 6,000 characters on paragraph boundaries. Cards are allotted to each chunk by its length, the chunks are sent to Claude in parallel (4 at a time), and the results are merged in document order with duplicate questions removed. A chunk that fails validation twice is skipped with a warning, while an API key or rate-limit error stops the whole run.
- **API key and cost control:** the key lives in Streamlit secrets and is never committed, and each request is capped at 40 cards and 10 chunks (about 60,000 characters).

## Results

Generated 10 cards from 1 page of HST 102 notes in about 15 seconds, versus about 15 minutes by hand.

## Limitations and next steps

- Notes longer than about 60,000 characters are cut off at 10 chunks.
- There is no OCR, so scanned slides can't be read.
- Card types are sometimes mislabeled.
- A quiz mode and spaced repetition are planned.

## Run locally

Requires Python 3.10+ and an [Anthropic API key](https://platform.claude.com/).

```bash
git clone https://github.com/Kerem-Oncer/lecture-flashcards.git
cd lecture-flashcards
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-key-here"
```

Then start the app:

```bash
streamlit run app.py
```
