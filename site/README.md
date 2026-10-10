# MultINav project page

Static project page for *Interactive Navigation Beyond Reachability: Benchmark and Agentic Framework*.
The layout follows the lab's WEM project page (<https://zgca-hmi-lab.github.io/WEM/>), and the page is
self-contained: no CDN, no build step, no external fonts.

## Local preview

```powershell
cd "C:\ldl\北京中关村学院 ZGCA\papers\MultINav_site"
python serve.py          # http://127.0.0.1:8811/  (port optional)
```

`serve.py` supports HTTP byte ranges, so the MP4 clips can be scrubbed in the browser. Opening
`index.html` directly also works, but seeking inside the larger videos is unreliable over `file://`.

## Layout

```
index.html                     page markup
serve.py                       local preview server with range support
static/css/index.css           all styling
static/js/index.js             scroll-spy, BibTeX copy, single-video playback
static/images/                 figures exported from the manuscript PDF
static/images/posters/         first-view posters extracted from each clip
static/videos/                 8 clips copied from papers/supplements
static/pdfs/multinav.pdf       manuscript PDF (local copy)
```

## Content sources

- Text, tables, and numbers: `papers/MultINav_HMI` (`Sections/`, `Tables/`, `main.pdf`).
- Figures: `papers/MultINav_HMI/Images/*.pdf`, rendered to PNG with Poppler at 200 dpi and trimmed.
- Videos: `papers/supplements/Exp-MulitINav/` (6 simulated ObjectGoal rollouts) and
  `papers/supplements/Exp-Real world/` (2 real-world rollouts). File names were normalized; the
  mapping is by original order and target object.
- Posters: frames captured at 40% of each clip.

## Figure and table order

Every numbered figure and table carries the number it has in the manuscript, and the page presents them
in the same sequence as the compiled paper:

| Paper item | On this page |
| --- | --- |
| Figure 1 | Teaser under the hero |
| Table 1 | Top of the MultINav-Bench section |
| Figure 2 | MultINav-Bench construction figure |
| Figure 3 | MultINav-Agent framework figure |
| Table 2 | First table in Results |
| Table 3 | Second table in Results |
| Table 4 | Third table in Results |
| Figure 4 | Case-study figure, above the simulated clips |
| Figure 5 | Real-world figure, above the real-world clips |

The five-metric definition table in the MultINav-Bench section and the qualitative video cards are page
extras; they are deliberately not numbered, because the paper presents that material as prose and as
supplementary media rather than as numbered floats.

## Known placeholders

The hero "Code" and "Dataset" buttons are intentionally non-clickable until public URLs exist, and
the BibTeX entry has no arXiv identifier yet. The footer keeps the Academic Project Page Template
attribution required by the template license.
