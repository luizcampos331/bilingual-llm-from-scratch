"""Generate article covers: a page from an engineer's lab notebook.

Visual rules shared by every cover in the series:
- cream graph paper, monospace "printed" content, handwritten annotations;
- the mission number is always written large on the left;
- a single accent color per language: amber for English, teal for Portuguese.

Usage:
    python tools/covers/make_cover.py 00

Writes cover-en.svg/png and cover-pt.svg/png into the mission's article folder.
PNG rendering uses headless Google Chrome (fonts are loaded from Google Fonts).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

W, H = 1600, 900
PAPER = "#f6f2e9"
GRID = "#e3ddcf"
MARGIN = "#c9cfd8"
INK = "#1e293b"
FAINT_INK = "#64748b"
LANG_COLOR = {"en": "#d97706", "pt": "#0d9488"}  # amber / teal, darkened for contrast on paper

HAND = "'Caveat', 'Bradley Hand', 'Noteworthy', cursive"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, monospace"
FONTS = ("@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@700"
         "&amp;family=JetBrains+Mono:wght@500;700&amp;display=block');")

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO_ROOT = Path(__file__).resolve().parents[2]


def page(content: str) -> str:
    grid = [f'<line x1="{x}" y1="0" x2="{x}" y2="{H}"/>' for x in range(0, W + 1, 32)]
    grid += [f'<line x1="0" y1="{y}" x2="{W}" y2="{y}"/>' for y in range(0, H + 1, 32)]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<style>{FONTS}</style>
<rect width="{W}" height="{H}" fill="{PAPER}"/>
<g stroke="{GRID}" stroke-width="1">{"".join(grid)}</g>
<line x1="140" y1="0" x2="140" y2="{H}" stroke="{MARGIN}" stroke-width="3"/>
{content}
</svg>'''


def mission_number(number: str, color: str) -> str:
    stroke = f'fill="none" stroke="{color}" stroke-linecap="round"'
    return (f'<text x="190" y="330" font-family="{HAND}" fill="{color}" font-size="230" '
            f'font-weight="700">{number}</text>'
            f'<path d="M 188 362 C 260 352, 340 366, 420 356" {stroke} stroke-width="6"/>'
            f'<path d="M 200 380 C 270 372, 330 382, 400 374" {stroke} stroke-width="4"/>')


# --- Mission 00: the roadmap itself is the artifact --------------------------------

MISSIONS_00 = {
    "en": ["manifesto", "bigram", "micrograd", "first neural net", "BPE tokenizer", "attention",
           "GPT", "pretraining", "modern architecture", "scaling", "fine-tuning", "release"],
    "pt": ["manifesto", "bigrama", "micrograd", "rede neural", "tokenizer BPE", "attention",
           "GPT", "pretraining", "arquitetura moderna", "escala", "fine-tuning", "lançamento"],
}
NOTES_00 = {
    "en": {"start": "starts now", "gpu": "rented GPU", "ship": "open model!", "top": "25 weeks · EN + PT"},
    "pt": {"start": "começa agora", "gpu": "GPU alugada", "ship": "modelo aberto!", "top": "25 semanas · EN + PT"},
}


def mission_00(lang: str) -> str:
    color = LANG_COLOR[lang]
    notes = NOTES_00[lang]
    hand = f'font-family="{HAND}" fill="{color}" font-weight="700"'
    stroke = f'fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"'

    checklist = []
    for i, name in enumerate(MISSIONS_00[lang]):
        x, y = (540, 1040)[i // 6], 230 + (i % 6) * 88
        done = i == 0
        checklist.append(f'<rect x="{x}" y="{y - 26}" width="30" height="30" rx="3" fill="none" '
                         f'stroke="{INK}" stroke-width="2.5"/>')
        checklist.append(f'<text x="{x + 52}" y="{y}" font-family="{MONO}" font-size="30" '
                         f'font-weight="{700 if done else 500}" fill="{INK if done else FAINT_INK}">'
                         f'{i:02d}  {name}</text>')
        if done:
            checklist.append(f'<path d="M {x + 2} {y - 12} L {x + 13} {y + 2} L {x + 40} {y - 42}" '
                             f'fill="none" stroke="{color}" stroke-width="7" stroke-linecap="round" '
                             f'stroke-linejoin="round"/>')

    annotations = [
        # "starts now" under the number, arrow into mission 00
        f'<text x="200" y="470" {hand} font-size="54" transform="rotate(-4 200 470)">{notes["start"]}</text>',
        f'<path d="M 432 236 C 460 222, 490 216, 522 216" {stroke}/>',
        f'<path d="M 502 202 L 524 216 L 504 232" {stroke}/>',
        # arrow and note next to mission 09
        f'<path d="M 1360 482 L 1300 482" {stroke}/>',
        f'<path d="M 1318 468 L 1298 482 L 1318 496" {stroke}/>',
        f'<text x="1372" y="496" {hand} font-size="46" transform="rotate(-3 1372 496)">{notes["gpu"]}</text>',
        # sketchy double circle around mission 11
        f'<ellipse cx="1210" cy="656" rx="178" ry="40" transform="rotate(-2 1210 656)" {stroke} stroke-width="3.5"/>',
        f'<ellipse cx="1214" cy="652" rx="184" ry="44" transform="rotate(1 1214 652)" {stroke} stroke-width="2"/>',
        f'<text x="1070" y="790" {hand} font-size="44" transform="rotate(-3 1070 790)">{notes["ship"]}</text>',
        f'<path d="M 1150 752 C 1160 730, 1170 716, 1180 700" {stroke} stroke-width="3"/>',
        # context note in the top-right corner
        f'<text x="1560" y="110" text-anchor="end" {hand} font-size="46" '
        f'transform="rotate(-2 1560 110)">{notes["top"]}</text>',
    ]
    return page(mission_number("00", color) + "".join(checklist) + "".join(annotations))


COVERS = {"00": ("00-manifesto", mission_00)}


def render_png(svg: str, png_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "cover.html"
        html.write_text(f'<html><body style="margin:0">{svg}</body></html>', encoding="utf-8")
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=5000",
             f"--window-size={W},{H}", f"--screenshot={png_path}", html.as_uri()],
            check=True, capture_output=True,
        )


def main(mission: str) -> None:
    folder, build = COVERS[mission]
    out_dir = REPO_ROOT / "articles" / folder
    for lang in LANG_COLOR:
        svg = build(lang)
        (out_dir / f"cover-{lang}.svg").write_text(svg, encoding="utf-8")
        render_png(svg, out_dir / f"cover-{lang}.png")
        print(f"wrote {out_dir / f'cover-{lang}.png'}")


if __name__ == "__main__":
    main(sys.argv[1])
