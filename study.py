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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap">
<style>
  :root {
    --card-bg: #ffffff; --text: #1c1b22; --muted: #6e6a7c; --border: #e4e0d7;
    --accent: #4f46e5; --accent-soft: rgba(79, 70, 229, 0.10); --on-accent: #ffffff;
    --shadow: 0 1px 2px rgba(28, 27, 34, 0.06), 0 12px 32px rgba(28, 27, 34, 0.10);
    --button-bg: #f3f1ec; --track: #ebe8e1;
  }
  :root[data-theme="dark"] {
    --card-bg: #1c1b25; --text: #eceaf4; --muted: #a09cb0; --border: #2d2b3a;
    --accent: #818cf8; --accent-soft: rgba(129, 140, 248, 0.16); --on-accent: #121118;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.4), 0 12px 32px rgba(0, 0, 0, 0.45);
    --button-bg: #24222f; --track: #2d2b3a;
  }
  html, body {
    margin: 0; background: transparent; color: var(--text);
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 640px; margin: 0 auto; padding: 8px 4px; box-sizing: border-box; }
  .scene { perspective: 1200px; height: 290px; border-radius: 18px; outline: none; }
  .scene:focus-visible { box-shadow: 0 0 0 3px var(--accent-soft), 0 0 0 1px var(--accent); }
  .card {
    position: relative; width: 100%; height: 100%; cursor: pointer;
    transform-style: preserve-3d; transition: transform 0.45s cubic-bezier(0.2, 0.7, 0.2, 1);
  }
  .card.flipped { transform: rotateY(180deg); }
  .card.no-animation { transition: none; }
  .face {
    position: absolute; inset: 0; box-sizing: border-box; padding: 48px 28px 28px;
    display: flex; flex-direction: column; overflow-y: auto; text-align: center;
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 18px;
    box-shadow: var(--shadow); backface-visibility: hidden; -webkit-backface-visibility: hidden;
  }
  .back { transform: rotateY(180deg); border-top: 4px solid var(--accent); }
  .content { margin: auto 0; }  /* Centers short text without clipping long text. */
  .label {
    position: absolute; top: 14px; left: 16px; padding: 3px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--accent); background: var(--accent-soft);
  }
  .text {
    font-family: "Fraunces", Georgia, "Times New Roman", serif; font-weight: 500;
    font-size: clamp(19px, 4.6vw, 25px); line-height: 1.4;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .source {
    margin: 18px auto 0; max-width: 90%; padding-left: 12px; text-align: left;
    border-left: 3px solid var(--border); font-size: 13.5px; line-height: 1.45;
    font-style: italic; color: var(--muted);
  }
  .bar { height: 4px; margin-top: 16px; border-radius: 999px; background: var(--track); overflow: hidden; }
  .bar-fill { height: 100%; width: 0; background: var(--accent); border-radius: 999px; transition: width 0.3s ease; }
  .controls {
    display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 12px;
  }
  .counter { min-width: 72px; text-align: center; color: var(--muted); font-variant-numeric: tabular-nums; }
  button {
    font: inherit; font-size: 14px; font-weight: 500; min-height: 38px; padding: 6px 14px; cursor: pointer;
    color: var(--text); background: var(--button-bg); border: 1px solid var(--border); border-radius: 10px;
    transition: border-color 0.15s ease, transform 0.1s ease;
  }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button:active:not(:disabled) { transform: translateY(1px); }
  button:disabled { opacity: 0.5; cursor: default; }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  #next { color: var(--on-accent); background: var(--accent); border-color: var(--accent); }
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
  <div class="bar" aria-hidden="true"><div class="bar-fill" id="bar-fill"></div></div>
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
    el("source").textContent = current.source_quote ? "“" + current.source_quote + "”" : "";
    el("counter").textContent = (position + 1) + " / " + CARDS.length;
    el("bar-fill").style.width = ((position + 1) / CARDS.length * 100) + "%";
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
