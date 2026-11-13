import re
from io import BytesIO

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from exporters import clean_cards, to_apkg, to_csv
from generator import MAX_CARDS, MAX_CHUNKS, MAX_INSTRUCTION_CHARS, chunk_text, generate_cards
from study import build_study_html

# Current Claude model IDs (docs.claude.com models overview, checked 2026-11-09):
#   Haiku 5.5:  "claude-haiku-5-5"   from $0.10 input / $0.50 output per million tokens
#   Sonnet 5.5: "claude-sonnet-5-5"  $2 input / $10 output per million tokens

PREVIEW_CHARS = 1500
STUDY_VIEW_HEIGHT = 420  # px; fits the card, buttons and hint without cutting anything off.
MIN_CHARS_PER_PAGE = 30  # Below this average, a PDF is probably scanned images.
CARD_TYPES = ["definition", "concept", "application"]
DEFAULT_DECK_NAME = "Lecture Flashcards"
AUTO_MODE = "Auto (cover every topic)"
CUSTOM_MODE = "Custom number"


@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Return a PDF's extracted text and its page count."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip(), len(pages)


def to_markdown(value) -> str:
    """Escape card text so symbols like * and $ display literally."""
    if pd.isna(value):
        return ""
    return re.sub(r"([\\`*_$~\[\]<>:#|])", r"\\\1", str(value).strip())


def plural(count: int, singular: str, plural_form: str) -> str:
    return f"{count:,} {singular if count == 1 else plural_form}"


def format_run_stats(result) -> str:
    """One line of run stats: time, chunks, calls, tokens and estimated cost."""
    counts = result.call_counts
    cost = result.estimated_cost
    cost_text = f"≈ \\${cost:.3f}" if cost >= 0.001 else "< \\$0.001"  # \$ avoids LaTeX.
    return " · ".join(
        [
            f"{result.total_seconds:.1f}s",
            plural(result.chunk_count, "chunk", "chunks"),
            f"{plural(sum(counts.values()), 'call', 'calls')} "
            f"({plural(counts['follow-up'], 'follow-up', 'follow-ups')}, "
            f"{plural(counts['retry'], 'retry', 'retries')})",
            plural(len(result.cards), "card", "cards"),
            f"{result.input_tokens:,} in / {result.output_tokens:,} out tokens "
            f"({round(result.output_tokens / max(len(result.cards), 1)):,}/card)",
            cost_text,
        ]
    )


st.set_page_config(page_title="Lecture to Flashcards", page_icon="🗂️")
st.title("Lecture → Flashcards")
st.caption(f"Uses the developer’s API key — max {MAX_CARDS} cards per request.")

with st.sidebar:
    st.header("Settings")
    course_name = st.text_input("Course name", placeholder="e.g. CSE 340")
    card_count_mode = st.radio("Card count", [AUTO_MODE, CUSTOM_MODE])
    st.caption(
        "Claude lists every topic in your notes, then writes one card per topic "
        f"(max {MAX_CARDS})."
    )
    auto_mode = card_count_mode == AUTO_MODE
    num_cards = MAX_CARDS
    if not auto_mode:
        num_cards = st.slider("Number of cards", min_value=5, max_value=MAX_CARDS, value=10)
    difficulty = st.selectbox("Difficulty", ["intro", "exam-level"])
    custom_instructions = st.text_area(
        "Custom instructions (optional)",
        max_chars=MAX_INSTRUCTION_CHARS,
        placeholder='e.g. "Focus on dates and key terms" or "Only cover deadlock and race conditions"',
    )

paste_tab, upload_tab, youtube_tab = st.tabs(
    ["Paste text", "Upload PDF/TXT", "YouTube transcript (paste)"]
)

with paste_tab:
    pasted_text = st.text_area("Lecture notes", height=250)

with upload_tab:
    uploaded_file = st.file_uploader("PDF or TXT file", type=["pdf", "txt"])
    uploaded_text = ""
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".pdf"):
            try:
                uploaded_text, page_count = extract_pdf_text(file_bytes)
            except Exception as error:
                st.error(f"Could not read this PDF: {error}")
            else:
                if len(uploaded_text) < MIN_CHARS_PER_PAGE * page_count:
                    st.warning(
                        "This PDF yielded almost no text. It may be scanned images, "
                        "which this app can't read."
                    )
        else:
            # utf-8-sig also strips the byte-order mark some Windows editors add.
            uploaded_text = file_bytes.decode("utf-8-sig", errors="replace")

with youtube_tab:
    transcript_text = st.text_area("Transcript", height=250)

sources = {
    "Paste text": pasted_text.strip(),
    "Upload PDF/TXT": uploaded_text.strip(),
    "YouTube transcript (paste)": transcript_text.strip(),
}
filled_tabs = [name for name, text in sources.items() if text]

# If several tabs have content, the first one in tab order wins.
active_tab = filled_tabs[0] if filled_tabs else None
st.session_state["source_text"] = sources[active_tab] if active_tab else ""

source_text = st.session_state["source_text"]
if source_text:
    if len(filled_tabs) > 1:
        st.info(f"Several tabs have content. Using: {active_tab}.")
    st.write(f"**{len(source_text):,} characters** from {active_tab}")
    with st.expander(f"Preview (first {PREVIEW_CHARS:,} characters)"):
        st.text(source_text[:PREVIEW_CHARS])
    chunk_count = len(chunk_text(source_text))
    if chunk_count > MAX_CHUNKS:
        st.info(
            f"These notes split into {chunk_count} chunks. Only the first {MAX_CHUNKS} "
            "will be used, to keep each request's cost down."
        )

if st.button("Generate flashcards", disabled=not source_text):
    progress = st.progress(0.0, text="Splitting notes into chunks...")
    live_preview = st.empty()
    cards_so_far = []

    def show_progress(done: int, total: int, chunk_cards: list) -> None:
        label = "chunk" if total == 1 else "chunks"
        progress.progress(done / total, text=f"Generated cards for {done} of {total} {label}...")
        cards_so_far.extend(chunk_cards)
        if cards_so_far:
            # Rebuilt after each chunk; cleared once the editable table is ready.
            with live_preview.container(height=320, border=True):
                st.caption(f"{len(cards_so_far)} cards so far")
                st.markdown(
                    "\n\n".join(
                        f"**Q:** {to_markdown(card.front)}  \n**A:** {to_markdown(card.back)}"
                        for card in cards_so_far
                    )
                )

    with st.spinner("Generating flashcards..."):
        try:
            result = generate_cards(
                source_text,
                num_cards,
                difficulty,
                course_name.strip(),
                custom_instructions=custom_instructions,
                auto=auto_mode,
                on_progress=show_progress,
            )
        except Exception as error:
            progress.empty()
            live_preview.empty()
            st.error(str(error))
        else:
            progress.empty()
            live_preview.empty()
            for warning in result.warnings:
                st.warning(warning)
            # Includes the topic column, which stays in session state but isn't shown.
            cards_df = pd.DataFrame([card.model_dump() for card in result.cards])
            st.session_state["cards"] = cards_df
            st.session_state["coverage"] = {
                "topics": result.topics,
                "covered": result.covered,
                "auto": auto_mode,
                "card_count": len(result.cards),
            }
            st.session_state["run_stats"] = format_run_stats(result)
            # The editor must keep the same input between reruns. Passing it its own
            # output would reset the widget and lose every second edit.
            st.session_state["cards_editor_input"] = cards_df.copy()
            st.session_state["cards_version"] = st.session_state.get("cards_version", 0) + 1

if "cards_editor_input" in st.session_state:
    st.subheader("Your cards")
    if "run_stats" in st.session_state:
        st.caption(st.session_state["run_stats"])
    coverage = st.session_state.get("coverage")
    if coverage and coverage["topics"]:
        topics, covered = coverage["topics"], coverage["covered"]
        n_covered = sum(topic in covered for topic in topics)
        if n_covered < len(topics):
            if coverage["auto"]:
                st.warning(f"Covered {n_covered} of {len(topics)} topics.")
            else:
                st.warning(
                    f"Found {len(topics)} topics; {coverage['card_count']} cards cover "
                    f"{n_covered} of them. Switch to Auto to cover all."
                )
        with st.expander(f"Topics found ({len(topics)})"):
            st.markdown(
                "  \n".join(
                    f"{':green[✓]' if topic in covered else ':red[✗]'} {to_markdown(topic)}"
                    for topic in topics
                )
            )

    study_tab, review_tab = st.tabs(["Study", "Review and edit"])

    # The Review tab's code runs first, even though Study is the first tab, so the
    # Study view is built from the edits made in this same rerun.
    with review_tab:
        edited_cards = st.data_editor(
            st.session_state["cards_editor_input"],
            key=f"cards_editor_{st.session_state['cards_version']}",
            num_rows="dynamic",
            hide_index=True,
            column_order=["front", "back", "card_type", "source_quote"],
            column_config={
                "front": st.column_config.TextColumn("Front", width="large", required=True),
                "back": st.column_config.TextColumn("Back", width="large", required=True),
                "card_type": st.column_config.SelectboxColumn(
                    "Type", width="small", options=CARD_TYPES, required=True
                ),
                "source_quote": st.column_config.TextColumn("Source quote", width="medium"),
            },
        )
        st.session_state["cards"] = edited_cards
        st.caption(f"{len(edited_cards)} cards")

        with st.expander("Card preview"):
            preview_cards = [
                card for card in edited_cards.to_dict("records") if to_markdown(card["front"])
            ]
            for position, card in enumerate(preview_cards):
                if position:
                    st.divider()
                st.markdown(f"**Q:** {to_markdown(card['front'])}")
                st.markdown(f"**A:** {to_markdown(card['back'])}")
                quote = to_markdown(card["source_quote"])
                if quote:
                    st.caption(f"Source: {quote}")

    with study_tab:
        if clean_cards(st.session_state["cards"]).empty:
            st.info("No complete cards to study. Give each card a front and a back in the Review tab.")
        else:
            st.iframe(
                build_study_html(st.session_state["cards"], dark=st.context.theme.type == "dark"),
                height=STUDY_VIEW_HEIGHT,
                alt="Flashcard study view",
            )

export_cards = clean_cards(st.session_state.get("cards", []))
no_cards = export_cards.empty
deck_name = course_name.strip() or DEFAULT_DECK_NAME
safe_file_name = re.sub(r'[\\/:*?"<>|]+', "_", deck_name)

st.subheader("Export")
apkg_column, csv_column = st.columns(2)
with apkg_column:
    st.download_button(
        "Download Anki deck (.apkg)",
        data=b"" if no_cards else to_apkg(export_cards, deck_name),
        file_name=f"{safe_file_name}.apkg",
        mime="application/octet-stream",
        disabled=no_cards,
    )
with csv_column:
    st.download_button(
        "Download CSV",
        data=b"" if no_cards else to_csv(export_cards),
        file_name=f"{safe_file_name}.csv",
        mime="text/csv",
        disabled=no_cards,
    )
