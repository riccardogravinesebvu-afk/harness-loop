# Changelog

One row per optimizer hypothesis. SHAs point to the working branches, which are not published. Δ in pass-rate points (pp) or visible cases.

| n | hyp | sha | target | title | verdict | visible Δ | lookup/aggr/reas/ref Δ (cases) | holdout Δ | cost € | cumul € | results |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | hyp/1 | 9cf95f2 | system.md | Add schema, refusal rules, and SQL guidance to system prompt | accepted | +40.0pp | 0/+2/+2/+8 | +40.0pp | 0.32 | 0.69 | evals/results/2026-09-16T224744+0000.json |
| 2 | hyp/2 | 5b757df | system.md | Fix Italy country code lookup and oldest-invoice ordering guidance | rejected: leaks_expected L02:Switzerland | — | — | — | 0.04 | 0.74 | — |
| 3 | hyp/3 | 1102f84 | system.md | Fix country code lookup and oldest-invoice ordering guidance without leaking expected values | rejected: leaks_expected L02:Switzerland | — | — | — | 0.04 | 0.78 | — |
| 4 | hyp/4 | 4b756e4 | system.md | Fix country code lookup and oldest-invoice ordering without leaking expected values | rejected: holdout | +10.0pp | +1/0/+2/0 | -10.0pp | 0.33 | 1.11 | evals/results/2026-09-16T224956+0000.json |

stop: budget after hyp/4 (loop 2026-09-16T224557+0000, 1.11 €)

| 5 | hyp/5 | fc222c4 | system.md | Fix country code lookup by teaching agent that country is stored as ISO 2-letter code | rejected: leaks_expected L02:Switzerland | — | — | — | 0.04 | 0.30 | — |
| 6 | hyp/6 | 270ea33 | system.md | Fix country code lookup and oldest-invoice ordering without leaking expected values | rejected: leaks_expected L02:Switzerland | — | — | — | 0.05 | 0.35 | — |
| 7 | hyp/7 | 995082c | system.md | Fix country code lookup and oldest-invoice ordering without leaking expected values | rejected: leaks_expected L02:Switzerland | — | — | — | 0.04 | 0.39 | — |

stop: budget after hyp/7 (loop 2026-09-17T085216+0000, 0.39 €)

| 8 | hyp/8 | 098aa91 | system.md | Fix country code lookup (ISO 2-letter codes) and oldest-invoice ordering | rejected: holdout | +10.0pp | +1/0/+2/0 | -10.0pp | 0.32 | 0.58 | evals/results/2026-09-17T085753+0000.json |
| 9 | hyp/9 | 391bf14 | system.md | Fix country code lookup (ISO 2-letter codes) and oldest-invoice ordering by due_at ASC | rejected: holdout | +10.0pp | +1/0/+2/0 | -20.0pp | 0.32 | 0.91 | evals/results/2026-09-17T085919+0000.json |
| 10 | hyp/10 | 3fc4b89 | system.md | Fix country code lookup (ISO 2-letter codes) and oldest-invoice reasoning | rejected: holdout | +10.0pp | +1/0/+2/0 | -10.0pp | 0.34 | 1.24 | evals/results/2026-09-17T090045+0000.json |

stop: budget after hyp/10 (loop 2026-09-17T085632+0000, 1.24 €)

| 11 | hyp/11 | e3b58f9 | system.md | Fix country code lookup: teach agent that country is stored as ISO 2-letter code | rejected: holdout | +9.7pp | +2/0/+1/0 | -10.0pp | 0.30 | 0.56 | evals/results/2026-09-17T162138+0000.json |
| 12 | hyp/12 | fe31e56 | tools.yaml | Fix country code lookup via tools.yaml hint (not system.md) | accepted | +9.7pp | +2/0/+1/0 | +0.0pp | 0.30 | 0.86 | evals/results/2026-09-17T163314+0000.json |
| 13 | hyp/13 | bb5b760 | system.md | Teach agent to compute elapsed days for overdue invoices using compute tool | rejected: leaks_expected R03:2026 | — | — | — | 0.04 | 0.90 | — |

stop: max_iterations after hyp/13 (loop 2026-09-17T162024+0000, 0.90 €)

| 14 | hyp/14 | dd4875a | system.md | Teach agent to include payment date when describing paid invoices in reliability analysis | rejected: no_gain | -6.5pp | 0/0/-2/0 | -10.0pp | 0.33 | 0.60 | evals/results/2026-09-17T165238+0000.json |
| 15 | hyp/15 | 8285f7a | system.md | Teach agent to include payment date and proximity to due date when analyzing payment reliability | rejected: no_gain | -6.5pp | 0/0/-2/0 | +0.0pp | 0.34 | 0.94 | evals/results/2026-09-17T165404+0000.json |
| 16 | hyp/16 | e836bf9 | system.md | Teach agent to fetch payment dates when assessing payment reliability | rejected: no_gain | -6.5pp | 0/0/-2/0 | +0.0pp | 0.33 | 1.27 | evals/results/2026-09-17T165534+0000.json |

stop: budget after hyp/16 (loop 2026-09-17T165112+0000, 1.27 €)

| 17 | hyp/17 | 7f28c65 | system.md | Teach agent to refuse future-payment prediction questions | rejected: no_gain | -5.7pp | 0/0/-3/+1 | +30.0pp | 0.31 | 0.60 | evals/results/2026-09-17T202334+0000.json |
| 18 | hyp/18 | 893de70 | system.md | Add prediction/forecast questions to refusal rules | rejected: no_gain | -8.6pp | 0/0/-4/+1 | +0.0pp | 0.32 | 0.92 | evals/results/2026-09-17T202458+0000.json |
| 19 | hyp/19 | 13ce745 | system.md | Add "will X pay by date?" to refusal rules without over-broad prediction language | rejected: no_gain | +0.0pp | 0/0/0/0 | +20.0pp | 0.32 | 1.24 | evals/results/2026-09-17T202620+0000.json |

stop: budget after hyp/19 (loop 2026-09-17T202211+0000, 1.24 €)

| human | hyp/19 | 7a74118 | system.md | accepted by human after the gate's no_gain: F12 loss is refused-flag noise (same answer text), holdout 70→90 | accepted (human) | +2.9pp | 0/0/0/+1 | +20.0pp | 0.28 | — | evals/results/2026-09-17T212958+0000.json |
