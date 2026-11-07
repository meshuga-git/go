# Working in this repository

- `docs/TAVNIT_Architecture_and_Refactoring_Plan.docx` is the
  authoritative plan. `PROGRESS.md` records each task's status, what
  was agreed in conversation and what the work has found. Read both
  before starting a phase, and update `PROGRESS.md` in every commit
  that finishes a task.
- One branch `phase-<n>` and one pull request into `main` per phase;
  one commit per task.
- Every change keeps `python tests/golden.py --toolchain linux-gcc`
  passing (byte-identical outputs) unless a decision record or a plan
  task says the output changes. Re-record golden files only in the
  commit that makes such a change, and name the reason in its message.
- Style: Google-style docstrings that state the minimum Python version
  the same way in every module; 79-character code lines; 72-character
  docstrings and comments; American English; Doxygen `/** @brief */`
  comments in C++. The code uses nothing newer than Python 3.10.
- Keep the user's decisions: Tjala stays unseeded with no seed option;
  the 65,000-gene cap stays; each script keeps its arguments, defaults,
  sentinels and configuration mechanisms; no new prompts for the
  fill-in policy or the sizes.
