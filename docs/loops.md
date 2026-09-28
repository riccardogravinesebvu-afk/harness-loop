# The loops, one by one

Six loops, nineteen hypotheses, about €8.7 of API spend. Each loop wrote its own summary file in
`evals/results/loops/`, each evaluated iteration wrote a results file, and every row here links to
one of them. The short version lives in the [README](../README.md); this is the blow-by-blow.

## Baseline

The starting prompt is one sentence and the tool descriptions are one line each. The agent doesn't
know the schema, so it explores with SQL until it hits the step cap, answers in prose instead of
calling `final_answer`, and never refuses anything.

45% total, 47% visible, 40% holdout, $0.42 a run.
([`2026-09-16T143601`](../evals/results/2026-09-16T143601+0000.json), seed 42, sha `75ee5e3`.)

I ran the same prompts twice before committing the code. Identical, case by case, except one failure
that hit the step cap instead of rambling.

## Loop 1

Budget stop after four hypotheses, €1.11.
([`2026-09-16T224557`](../evals/results/loops/2026-09-16T224557+0000.json))

| n | hypothesis | verdict | visible | holdout | per category (visible) | € |
|---|---|---|---|---|---|---|
| 0 | baseline | | 50% | 40% | 86 / 71 / 50 / 0 | 0.37 |
| 1 | schema, refusal rules, SQL guidance in `system.md` | **accepted** | **90%** | **80%** | 86 / 100 / 75 / 100 | 0.32 |
| 2 | ISO country codes and ordering rules | rejected before eval: prompt spells out `Switzerland` | | | | 0.04 |
| 3 | same, reworded | rejected before eval: same literal | | | | 0.04 |
| 4 | same, literal removed | rejected: holdout fell | 100% | 70% | 100 / 100 / 100 / 100 | 0.33 |

Hypothesis 1 is the whole curve. The optimizer read `ledger.py` and `tools.py`, worked out that the
agent had no idea what the schema was, and wrote the schema, the outstanding/overdue arithmetic, the
refusal policy and "always call `final_answer`" into the system prompt. Refusals went from 0 to 100%
in one shot.

Then three failures in a row, which is why the budget rule fired.

## Loops 2 and 3

Nothing accepted. €0.39 and €1.24.
([`085216`](../evals/results/loops/2026-09-17T085216+0000.json),
[`085632`](../evals/results/loops/2026-09-17T085632+0000.json);
baselines [`085218`](../evals/results/2026-09-17T085218+0000.json) and
[`085633`](../evals/results/2026-09-17T085633+0000.json), 87% visible and 80% holdout both times)

| loop | n | hypothesis | verdict | visible | holdout |
|---|---|---|---|---|---|
| 2 | 5, 6, 7 | ISO country codes, three wordings | rejected before eval, anti-leak | | |
| 3 | 8 | ISO codes plus overdue-invoice reasoning | rejected: holdout | 97% | 70% |
| 3 | 9 | same, reworded | rejected: holdout | 97% | 60% |
| 3 | 10 | same, reworded | rejected: holdout | 97% | 70% |

Loop 2 went entirely to a scope bug in my own check. The optimizer wanted "`'CH'` for Switzerland" in
a list of ISO codes, and `Switzerland` is the accepted answer of `L02`, a case that was already
passing. The anti-leak rule refused all three wordings without running an eval. Five rejections
across the two loops, €0.21, no measurement. The check now considers only currently-failing cases,
which are the only ones whose expected value the optimizer is shown.

Loop 3 is the same idea three more times, all rejected on the holdout. The initial reading was that
the gate had caught one overfit four times over. The replications below complicate that: the case
that keeps falling, `F08`, also flips on its own, so the signal is part regression and part noise,
and at one run per decision the gate could not separate them.

The per-category gate, the thing I built first, never fired once. Not here, not in any later loop.
It is covered by unit tests and nothing else.

## Loop 4, with feedback

Max-iterations stop, €0.90.
([`162024`](../evals/results/loops/2026-09-17T162024+0000.json),
baseline [`162025`](../evals/results/2026-09-17T162025+0000.json))

Before starting it I flagged a failing run through the feedback endpoint: `L09`, "List the customers
based in Italy", failing since loop 1, with the note "the answer lists nobody; country is stored as a
code, not a name". That doubles the case's weight, so the weighted visible baseline reads 84% instead
of 87%, and the optimizer now sees my note sitting next to the failure.

| n | hypothesis | verdict | visible (weighted) | holdout | € |
|---|---|---|---|---|---|
| 0 | baseline, `L09` at weight 2 | | 84% | 80% | 0.26 |
| 11 | ISO country codes in `system.md` | rejected: holdout, lost `R06` | 94% | 70% | 0.30 |
| 12 | same hint, moved into the `run_sql` description | **accepted** | **94%** | 80% | 0.30 |
| 13 | days-overdue arithmetic | rejected before eval: anti-leak on `2026` | | | 0.04 |

**Two iterations to recover the flagged case.** Flagged before 11, passing in 12, which was accepted.
The `feedback` block of the loop file records both numbers.

Two things happened here that hadn't before. After the holdout rejection the optimizer stopped
rewording and moved the change to a different file, which is exactly what I'd added to its prompt
after loop 3. And the fix that had failed the holdout five times inside `system.md` passed as a
one-line tool description.

The second of those turns out to be sampling. See the replications below.

One more false positive from the same check: hypothesis 13 was refused for mentioning `2026`, a year
inside a date in one reference. Fixed.

## Variance after hypothesis 12

Three runs of the same prompts, same seed, back to back:
[`163541`](../evals/results/2026-09-17T163541+0000.json),
[`164030`](../evals/results/2026-09-17T164030+0000.json),
[`164128`](../evals/results/2026-09-17T164128+0000.json).

| metric | min | mean | max |
|---|---|---|---|
| total (weighted) | 85% | 87.0% | 88% |
| visible | 93.5% | 93.5% | 93.5% |
| holdout | 60% | 66.7% | 70% |
| reasoning | 60% | 66.7% | 70% |

The visible split is stable: the same 29 of 31 weighted cases pass every time. The holdout is not,
and this is where I found the hole in my own gate.

Hypothesis 12 was accepted on a run where the holdout read 80%, with `F08` passing. In all three
re-runs of the identical prompts `F08` fails: the agent declines in words but doesn't set
`refused=true`, which is the same regression the gate had rejected five times when the hint lived in
`system.md`. Moving it to a tool description didn't fix anything. One run happened to come up heads.
The honest holdout for those prompts is 67%, not 80%.

So: a gate with one-case resolution, one run of evidence, and one case of noise. A real regression
walked through it. The fix is in loop 5.

Four cases fail in all three runs: `F04`, `R03`, `R04`, `R09`. Reading the judge's reasoning on
`R03` and `R04`, it was penalising sums of figures the tools had returned separately: "5,250
outstanding" against a tool output of 3,750 and 1,500. A rubric property, not an agent property.
Addressed below.

## Loop 5, with holdout confirmation

Budget stop, nothing accepted, €1.27.
([`165112`](../evals/results/loops/2026-09-17T165112+0000.json),
baseline [`165113`](../evals/results/2026-09-17T165113+0000.json))

First, the fix: a hypothesis that passes the gate now gets the ten holdout cases re-run a second time
(about $0.10) and is kept only if that run holds too. The lower of the two becomes the bar for the
next comparison, so a lucky draw can't raise the bar behind you. Why: [DESIGN.md](../DESIGN.md#the-gate-per-category-on-held-out-cases-confirmed-twice).

| n | hypothesis | verdict | visible | holdout | € |
|---|---|---|---|---|---|
| 0 | baseline, re-measured | | 97% | 70% | 0.26 |
| 14 | payment dates in reliability analysis | no gain: fixed `R04`, broke four | 90% | 60% | 0.33 |
| 15 | same, reworded | no gain: fixed `R09`, broke three | 90% | 70% | 0.34 |
| 16 | same, reworded | no gain: fixed `F08`, broke three | 90% | 70% | 0.33 |

The confirmation never ran, because nothing passed the gate. It's exercised by
`tests/test_loop_mechanics.py` and nothing else.

What loop 5 actually showed is that the loop had hit the floor. With one or two visible failures
left, every candidate fix buys one case and costs two to four judged reasoning cases: more detail in
the prompt means more figures the judge decides are unsupported. Three flat iterations, budget stop,
correct behaviour all round. More iterations would not have helped. A better dataset and a better
judge would.

The optimizer also reworded the same idea three times after a `no_gain` verdict, which its prompt
already tells it not to do. My rule only covered holdout rejections.

## Fixing the measurement

Two changes, both committed before any new run (`eb049ca`), then a fresh baseline
([`202029`](../evals/results/2026-09-17T202029+0000.json)).

Four cases added, none changed, ever. `F11` and `F12` are visible refusals of the kind that was only
failing in the holdout, a prediction and an external fact, so the optimizer can finally see that
class of failure. `R11` and `R12` came from the optimizer's own proposals in `evals/proposed.yaml`,
which is where it can suggest cases that only a human may promote.

Judge v2 counts a figure as supported when it can be reached from the tool outputs by adding,
subtracting, dividing or counting. Every results file now carries `judge_version`.

| same prompts, same seed | 40 cases, judge v1 | 44 cases, judge v2 |
|---|---|---|
| total | 87% | 91% |
| visible | 94% | 97% |
| holdout | 67% | 70% |
| cases that flip across 3 runs | 1 | 0 |

The agent did not change. That jump is the rubric. It's also why numbers either side of this line
are never compared in the README.

Three runs on the new measurement
([`202749`](../evals/results/2026-09-17T202749+0000.json),
[`202856`](../evals/results/2026-09-17T202856+0000.json),
[`203003`](../evals/results/2026-09-17T203003+0000.json)) come out identical, 0 of 44 cases changing
outcome.

## Loop 6

Budget stop, nothing accepted by the loop, €1.24.
([`202211`](../evals/results/loops/2026-09-17T202211+0000.json))

| n | hypothesis | verdict | visible | holdout | € |
|---|---|---|---|---|---|
| 0 | baseline, re-measured | | 97% | 70% | 0.29 |
| 17 | refuse future-payment predictions, long rule | no gain: fixed all four failures, broke three | 91% | 100% | 0.31 |
| 18 | prediction rule, broader | no gain: broke five reasoning cases | 89% | 70% | 0.32 |
| 19 | one line: "will X pay?" can't be answered from the ledger | no gain: fixed `F04`, `F08`, `F11`, lost `F12` | 97% | 90% | 0.32 |

Giving the optimizer a visible refusal worked immediately. `F11` was failing where it could see it,
so it went after prediction refusals for the first time in six loops, and by the third attempt it had
the minimal version: one line.

Hypothesis 19 fixes all three prediction and external-fact refusals, two of them held out, and the
holdout goes 70% to 90%. The gate rejected it, because the visible rate has to rise strictly and this
gains `F11` while losing `F12`. Net zero.

The gate is right. Accepting on holdout gain would turn the holdout into a target, which is the one
thing it can never be.

But `F12` is the noise case. Under hypothesis 19 the agent's answer to "How many employees does
Alpine Foods have?" is the same sentence it gave at baseline, "I cannot provide employee count
information", and only the `refused` flag flips. So I merged the branch by hand, re-ran the suite,
and wrote it in the changelog as `accepted (human)`:

98% total, 100% visible, 90% holdout
([`212958`](../evals/results/2026-09-17T212958+0000.json), commit `7a74118`).

The loop's own curve stops at hypothesis 19. The human decision is a separate row with its own
results file, so nobody has to guess which is which.

## Where it ends

One case still fails: `F08`, which refuses in the right words and sets the wrong flag. The check is
right to demand the flag, because a refusal the caller can't detect isn't a refusal.

The remaining levers are outside the loop: more cases, and a refusal signal less fragile than a
boolean the model has to remember to set.
