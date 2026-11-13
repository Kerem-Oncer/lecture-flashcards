# Lecture → Flashcards

Turn messy lecture notes, PDFs, and YouTube transcripts into source-grounded flashcards you can review, edit, and export to Anki.

**Live demo:** [Open the app](https://lecture-flashcards-ckr8hemndpmqnbx5tvz2eo.streamlit.app/)

![Study view: a generated HST 102 flashcard with flip, shuffle and keyboard navigation, and the sidebar with Auto mode and custom instructions](docs/screenshot.png)

**Built with:** Python, Streamlit, Claude API, pydantic, pypdf, genanki, pandas

## How it works

Extract text (pypdf) → LLM generates JSON (Claude API) → validated with pydantic, with one retry on error → human review in an editable table → export to Anki `.apkg` or CSV

## Design decisions

- **Source quotes for grounding:** the model may only use the provided notes, and each card includes a short quote from them, so answers can be checked against the source during review.
- **Schema validation with retry:** every response is validated against a pydantic schema. On failure, the exact error goes back to the model for one retry, and a second failure shows a clear error.
- **Planning before writing (Auto mode):** the model lists topics first, pydantic checks that every card maps to a listed topic, and the code sends one follow-up request for any topic without a card. Prompts alone didn't guarantee coverage, so coverage is verified in code.
- **Human review before export:** cards land in an editable table, so nothing reaches a study deck without a chance to fix or delete it.
- **Study in the browser:** a Study tab shows one card at a time with a 3D flip animation, Prev/Next and Shuffle buttons, and keyboard shortcuts (Space to flip, arrow keys to move), so you can study without exporting to Anki. Card text is untrusted (it comes from the LLM and uploaded notes), so it is passed to the page as escaped JSON and set with `textContent`, never `innerHTML`.
- **Stable genanki IDs:** a fixed note type ID, a deck ID hashed from the deck name, and note GUIDs based on the card front mean re-importing updates cards instead of duplicating them.
- **Chunking for long notes:** text is split on paragraph boundaries into the fewest chunks of up to about 3,000 characters, sized evenly, since a parallel run is only as fast as its largest chunk. Smaller chunks mean each request finishes sooner and more of them run at once, and replies are requested as minified JSON so no output tokens go to whitespace. Cards are allotted to each chunk by its length, the chunks are sent to Claude in parallel (up to 8 at a time), each chunk's cards appear in a live preview as soon as it finishes, and the results are merged in document order with duplicate questions removed. A chunk that fails validation twice is skipped with a warning, while an API key or rate-limit error stops the whole run.
- **Custom instructions that can't override grounding:** an optional 500-character box lets you steer focus, difficulty, and style (e.g. "only cover deadlock"). The instructions go in their own tagged block, and the system prompt ranks them below the core rules, so they can narrow which cards appear but can't bring in outside facts or drop source quotes. If nothing in the notes matches, the app returns fewer cards or a clear message instead of inventing content.
- **API key and cost control:** the key lives in Streamlit secrets and is never committed, and each request is capped at 20 chunks (about 60,000 characters). Auto mode allows up to 80 cards per request and Custom up to 40. Tags (course and topic) are built in code instead of by the model, which saves output tokens.
- **Measured runs:** every API call records its token usage, duration, and type (first, retry, or follow-up), and each run reports its total time, tokens, and estimated cost in one line under the results.

## Results

Generated 10 cards from 1 page of HST 102 notes in about 15 seconds, versus about 15 minutes by hand.

## Limitations and next steps

- Notes longer than about 60,000 characters are cut off at 20 chunks.
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
