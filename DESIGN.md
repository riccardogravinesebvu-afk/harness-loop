# Design

The loop is small; the decisions around it are what make its numbers mean anything. Each section
below is one choice, the reason for it, and what was rejected. The numbers they are judged by are
all in `evals/results/`.

## Three levels of evaluation, cheapest first

A hypothesis meets three filters, and each one only runs if the previous one passed.

1. **Static checks on the hypothesis, before any eval is spent.** A length cap per prompt file
   (6,000 characters for `system.md`, 3,000 for `tools.yaml`) and an anti-leak check that refuses a
   prompt spelling out the expected answer of a currently failing visible case. Both cost nothing
   and both have fired: six hypotheses were refused here.
2. **Deterministic checks per case.** `exact`, `contains`, `contains_any`, `number` (with a
   tolerance of 0.5% of the target) and `refused`, which reads the agent's structured flag.
   Three of the four categories are graded only this way.
3. **An LLM judge, only where no deterministic truth exists.** Reasoning answers go to a judge
   scoring against a reference a human wrote from the same data.

The order is the point: a deterministic check can't be flattered and costs nothing, so the judge
is the exception, not the default. The alternative, judging every answer, would make the whole
score depend on one model's reading and add a judge call to every case.

The leak check looks only at failing cases, because those are the only `expected` values the
optimizer is shown. Scanning every visible case refused an ISO-code list for containing
`Switzerland`, a word a passing case happened to expect.

## The judge: a weighted rubric with a penalty for invented numbers

`evals/judge.py` scores three dimensions from 0 to 1: correctness (0.5), grounding (0.3) and
completeness (0.2). A case passes at 0.7. If the answer contains a number that neither appears in
the tool outputs nor follows from them by adding, subtracting, dividing or counting, the whole
score is multiplied by 0.3.

- **Why weighted dimensions and not one score.** A single 0–10 grade moves with the judge's mood;
  three narrower questions are easier to answer consistently, and the rationale says which one
  failed.
- **Why a multiplicative penalty.** An additive one lets a well-written answer with one invented
  figure still pass. For a data agent, an invented number is the failure that matters most, so it
  sinks the score whatever the prose is like.
- **Why the judge sees the tool outputs.** Grounding can only be checked against what the agent
  actually retrieved, not against the reference alone.
- **Why the judge has a version.** Version 1 flagged any figure not literally present in the
  outputs, so a correct sum of two retrieved amounts failed. Version 2 accepts derived figures.
  Results files carry `judge_version`, and numbers from the two versions are never pooled.

## The gate: per category, on held-out cases, confirmed twice

A hypothesis is kept only if all of these hold (`src/optimizer/gates.py`):

- the visible pass rate rises;
- no category loses more than one visible case. The tolerance is `1/n` per category, computed
  from the dataset, so it adapts when cases are added;
- the holdout pass rate does not fall, on two separate runs. When the gate says yes, the loop
  re-runs only the holdout, and the lower of the two runs becomes the new bar.

**Why per category.** An aggregate score can rise while refusals collapse; a tolerance band on
the total can't see that trade.

**Why a held-out split the optimizer never sees.** Five of nineteen hypotheses improved the
visible cases and lowered the holdout. Without the split all five would have been kept.

**Why confirm twice.** With ten holdout cases one case is ten points, and measured run-to-run
noise was about one case. One hypothesis passed on a lucky run; three replications put it lower,
failing the same case each time. The second run costs about $0.10 per acceptance.

**Rejected:** a single run with a noise margin (the margin would be wider than any real gain at this
size). A paired test over several replications per hypothesis is the natural next step; this rule is
the cheapest one that would have caught the lucky run.

## What the optimizer sees and what it may touch

- It sees the current prompts, the agent's code (read only), the visible pass rate per category,
  every failing visible case with the agent's answer and tool calls, and the changelog of what was
  already tried. It never sees held-out cases. Through the changelog it does see when a hypothesis
  was rejected on the holdout and by how much, so the holdout is partly adaptive; the confirmation
  run below is the defence against that, and a fresh lockbox split would be the stronger one.
- It writes one of two files per iteration, `system.md` or `tools.yaml`, as a complete file.
  LLM-written diffs get whitespace wrong; a full file is robust and git shows the diff anyway.
- Its own prompt (`src/optimizer/prompt.md`) is generic and carries no knowledge of the domain.
  If it did, the improvement curve would measure the prompt author, not the optimizer.
- It may propose new eval cases. They land in `evals/proposed.yaml` and stay there until a human
  moves them: a loop that writes its own exam isn't measuring anything. Cases are added, never
  weakened.

## Cost budget and the stop rule

Cost per iteration is computed from token usage against a local price table
(`src/pricing.py`). It includes the optimizer's own tokens and every rejected hypothesis, because
a loop that only counts its successes looks cheaper than it is.

The loop stops when the last three iterations together bought less than 0.03 of visible pass rate
(three points) per euro (`src/optimizer/budget.py`). Gain is measured on the best accepted state,
so rejections add cost and no gain. Two hard limits sit behind it: 12 iterations and €10 in total.

**Why a marginal rule and not a fixed number of iterations.** The first missing piece of
knowledge is worth a lot and the tenth is worth little; the rule stops when the loop is paying for
noise. In practice it ended five of six loops, each after three flat iterations.

## Human feedback, weighted

`POST /feedback` takes a run's trace id, a verdict and a note (`src/feedback/api.py`).

- If the run was a dataset case, that case's weight doubles (`FEEDBACK_WEIGHT=2`) and the note is
  shown to the optimizer next to the failure.
- If it was an ad-hoc question, it becomes a new judged case with the note as its reference.
- The loop re-reads feedback at every iteration and records after how many iterations a flagged
  case came back.

**Why weight instead of a separate queue.** The pass rate the gate reads is already weighted, so a
doubled case costs twice as much when it fails; no second mechanism is needed.

A human can also accept a hypothesis the gate rejected, with a confirmation run committed and a
separate `accepted (human)` row in the changelog. The loop's own curve never includes it.

## Reproducibility

- The loop refuses to start on a dirty working tree.
- Each hypothesis gets its own branch. An accepted one fast-forwards `main`; a rejected one keeps
  its branch, and only the evidence is committed.
- Results files written by the current `evals/run.py` carry the git sha, the seed, the models,
  `judge_version` and a hash of the dataset, so two numbers are compared only when they measure the
  same thing. The committed ones predate some of these fields: judge-v1 files have no
  `judge_version`, and none carries the dataset hash.
- Temperature is 0 everywhere. The seed is passed to the provider, which may ignore it; measured
  variance is reported instead of assumed away.
