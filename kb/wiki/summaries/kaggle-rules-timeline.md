---
title: Kaggle Rules, Timeline and Prizes
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-rules.md, raw/articles/kaggle-timeline-prizes.md]
tags: [competition, rules, constraints]
---

# Summary — Rules, Timeline and Prizes

**Code-competition constraints (the load-bearing ones):**
- Submissions must go through Kaggle **Notebooks**; the "Submit" button requires: CPU **and** GPU run-time ≤ 9 hours, **internet disabled** at inference, and only freely & publicly available external data — including **pretrained models** — attached as Kaggle datasets.
- The whole solution (retrieval, scoring, de novo generation) must therefore run offline in ≤ 9 h; any external resource (COCONUT pool, pretrained weights) must be packaged as a dataset. See [[concepts/solution/Submission Architecture]].

**External data and tools:** allowed if publicly available and equally accessible to all participants at no cost, subject to a host-applied "Reasonableness Standard" (no excessive-cost LLMs/data/tools). Winning solutions require open-sourcing under an OSI license (with a carve-out for incompatible-licensed inputs). Public code sharing on the competition's Kaggle forum is permitted and licensed OSI.

**Submission limits:** 5 per day; single account per participant; team mergers capped by the cumulative daily quota.

**Timeline (11:59 PM UTC):**
- Start: Sep 14, 2026
- Entry deadline + team merger deadline: Dec 7, 2026
- Final submission deadline: **Dec 14, 2026**

**Prizes: $50,000 total** — 1st $16k, 2nd $12k, 3rd $9k, 4th $7k, 5th $6k.

**Acknowledgements:** CASMI (Critical Assessment of Small Molecule Identification) was founded 2012 by Emma Schymanski and Steffen Neumann, run 2012–2022 by academic teams; Enveda revives the name with permission ([[entities/CASMI]]).
