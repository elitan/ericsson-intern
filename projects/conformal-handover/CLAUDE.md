# Project: Conformal Handover

- always use `uv` for package management. never pip.
- use underscore_case for python files
- python source lives in src/handover/
- report is in report.tex (LaTeX, IEEE format)
- **README.md is the status log and todo tracker — keep it up to date**

## Cloud GPU

~$10 budget on vast.ai for larger experiments. See parent project's CLAUDE.md for vast.ai commands.

## Key concepts

- Handover = reassigning UE session from one cell to another
- Contextual bandit: state (UE measurements) → action (target cell) → reward
- CP provides prediction sets with coverage guarantees
- Adaptive protocol: small set → predictive HO, large set → measurement-based HO
