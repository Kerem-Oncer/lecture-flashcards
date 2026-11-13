"""Build the in-browser flashcard study view (a self-contained HTML/CSS/JS page)."""

import json

from exporters import clean_cards

# Fixed template: the only dynamic parts are the theme name and the escaped cards JSON.
# Card text is untrusted (it comes from the LLM and from uploaded notes), so the script
# sets it with textContent only, never innerHTML.
_TEMPLATE = """<!doctype html>
<html data-theme="__THEME__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --card-bg: #ffffff; --text: #262730; --muted: #6b6f7b; --border: #e3e4e8;
    --shadow: 0 6px 24px rgba(0, 0, 0, 0.10); --button-bg: #f0f2f6; --focus: #ff4b4b;
  }
  :root[data-theme="dark"] {
    --card-bg: #262730; --text: #fafafa; --muted: #a3a8b8; --border: #3d3f4b;
    --shadow: 0 6px 24px rgba(0, 0, 0, 0.45); --button-bg: #31333f;
  }
  html, body {
    margin: 0; background: transparent; color: var(--text);
    font-family: "Source Sans Pro", "Source Sans 3", system-ui, -apple-system, sans-serif;
  }
  .wrap { max-width: 640px; margin: 0 auto; padding: 8px 4px; box-sizing: border-box; }
  .scene { perspective: 1200px; height: 300px; border-radius: 16px; outline: none; }
  .scene:focus-visible { box-shadow: 0 0 0 2px var(--focus); }
  .card {
    position: relative; width: 100%; height: 100%; cursor: pointer;
    transform-style: preserve-3d; transition: transform 0.4s ease;
  }
  .card.flipped { transform: rotateY(180deg); }
  .card.no-animation { transition: none; }
  .face {
    position: absolute; inset: 0; box-sizing: border-box; padding: 36px 24px 24px;
    display: flex; flex-direction: column; overflow-y: auto; text-align: center;
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px;
    box-shadow: var(--shadow); backface-visibility: hidden; -webkit-backface-visibility: hidden;
  }
  .back { transform: rotateY(180deg); }
  .content { margin: auto 0; }  /* Centers short text without clipping long text. */
  .label {
    position: absolute; top: 12px; left: 16px; font-size: 12px; color: var(--muted);
    letter-spacing: 0.06em; text-transform: uppercase;
  }
  .text {
    font-size: clamp(18px, 4.5vw, 24px); line-height: 1.4;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .source { margin-top: 16px; font-size: 14px; line-height: 1.4; color: var(--muted); }
  .controls {
    display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 16px;
  }
  .counter { min-width: 72px; text-align: center; color: var(--muted); font-variant-numeric: tabular-nums; }
  button {
    font: inherit; font-size: 15px; min-height: 40px; padding: 6px 14px; cursor: pointer;
    color: var(--text); background: var(--button-bg); border: 1px solid var(--border); border-radius: 8px;
  }
  button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
  .shuffle { margin-left: auto; }
  .spacer { margin-right: auto; visibility: hidden; }
  .hint { margin-top: 10px; text-align: center; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <div class="scene" id="scene" tabindex="0" role="button" aria-label="Flashcard. Click or press Space to flip.">
    <div class="card" id="card">
      <div class="face front">
        <span class="label">Question</span>
        <div class="content"><div class="text" id="front"></div></div>
      </div>
      <div class="face back">
        <span class="label">Answer</span>
        <div class="content"><div class="text" id="back"></div><div class="source" id="source"></div></div>
      </div>
    </div>
  </div>
  <div class="controls">
    <button class="spacer" tabindex="-1" aria-hidden="true">Shuffle</button>
    <button id="prev" aria-label="Previous card">&larr; Prev</button>
    <span class="counter" id="counter" aria-live="polite"></span>
    <button id="next" aria-label="Next card">Next &rarr;</button>
    <button class="shuffle" id="shuffle">Shuffle</button>
  </div>
  <div class="hint">Click or press Space to flip &middot; &larr; &rarr; to move</div>
</div>
<script>
  const CARDS = __CARDS_JSON__;
  let order = CARDS.map((_, i) => i);
  let position = 0;
  const el = (id) => document.getElementById(id);
  const card = el("card");

  function showFront() {
    // Jump back to the front without animating, so the next card's answer never shows mid-flip.
    card.classList.add("no-animation");
    card.classList.remove("flipped");
    void card.offsetWidth;  // Apply the change before turning animation back on.
    card.classList.remove("no-animation");
  }

  function render() {
    showFront();
    const current = CARDS[order[position]];
    el("front").textContent = current.front;
    el("back").textContent = current.back;
    el("source").textContent = current.source_quote ? "Source: " + current.source_quote : "";
    el("counter").textContent = (position + 1) + " / " + CARDS.length;
  }

  function flip() { card.classList.toggle("flipped"); }
  function move(step) {
    position = (position + step + CARDS.length) % CARDS.length;
    render();
  }
  function shuffle() {
    for (let i = order.length - 1; i > 0; i--) {  // Fisher-Yates
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    position = 0;
    render();
  }

  el("scene").addEventListener("click", flip);
  el("prev").addEventListener("click", () => move(-1));
  el("next").addEventListener("click", () => move(1));
  el("shuffle").addEventListener("click", shuffle);
  document.addEventListener("keydown", (event) => {
    if (event.key === " ") { event.preventDefault(); flip(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); }
    else if (event.key === "ArrowRight") { event.preventDefault(); move(1); }
  });

  if (CARDS.length) {
    render();
  } else {
    el("front").textContent = "No cards to study yet.";
    for (const id of ["prev", "next", "shuffle"]) el(id).disabled = true;
  }
</script>
</body>
</html>
"""


def build_cards_json(cards_df) -> str:
    """Return the study cards as JSON that is safe to place inside a <script> tag.

    Skips rows with an empty front or back. Escaping every <, > and & as a \\u escape
    means card text can never close the script tag ("</script>") or open an HTML
    comment ("<!--"); JavaScript decodes the escapes back to the original characters.
    """
    cards = clean_cards(cards_df)
    records = [
        {"front": card["front"], "back": card["back"], "source_quote": card["source_quote"]}
        for card in cards.to_dict("records")
    ]
    return (
        json.dumps(records)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def build_study_html(cards_df, dark: bool = False) -> str:
    """Return the full study-view page for the given cards."""
    theme = "dark" if dark else "light"
    return _TEMPLATE.replace("__THEME__", theme).replace("__CARDS_JSON__", build_cards_json(cards_df))
