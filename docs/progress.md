# Implementation ledger: docs/superpowers/plans/2026-10-24-supermix-tether.md

- User delegated architecture choice and execution: `you decide`.
- Workspace initially empty, no existing Git repository or AGENTS.md found.
- Architecture ruling: select trainable causal-use receiver over another generic composition/risk/evidence subsystem; IIT is mandatory same-information baseline.
- Resource ruling: local CPU and installed dependencies; no cloud or upstream model downloads.
- Interface preflight: data supplies posterior tensor [1797,3,10]; batch expands to [n,3,3,10]; model consumes batches; runner owns all checkpoint/result formats; server reads runner artifacts.
- Research reviews: all 27 links assigned in three groups of nine; no upstream execution or mutation.
- Tasks 1, 2 and 4: data/donors, learned receiver and local viewer implemented with initial failing tests then passing focused suites.
- Task 3: runner/checkpoints/metric replay/strict inference replay implemented. Initial full suite passed 80 tests; regression and unpaired-control additions passed 86 tests.
- Independent review found and fixed inconsistent damage interventions, nonexclusive output reservation, and missing marginal-loss control. Also repaired large seed substream overflow and inference RNG side effects.
- Ruling: preserve legacy pipeline pilot at local commit a7283b3; do not treat its damaged-condition metrics or test-pool engineering pilot as independent confirmation.
- Ruling: add unpaired residual control and require its comparison before attributing a gain to correct pairing. Extra source code/controls chosen from methodological review, not test accuracies.
- Final protocol: docs/frozen-protocol.json. All seven arms use 400 receiver steps; weights remain 1/1 selected before pilot. Eight fresh training seeds 101-108; same underlying image partition.
- Completed: final independent fix review, frozen eight-seed confirmation study (`outputs/confirmation`), strict checkpoint/metric replay (`114` files, `4,312` metric checks, `2,352` inference checks), full empirical writeup (`docs/results.md`), and local server verification on `http://127.0.0.1:8797/` serving `outputs/confirmation`.
