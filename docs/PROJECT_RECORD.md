# The project in one file: every experiment, its setup, inputs, metric, timing and evidence

Written November 27, 2026. One entry per experiment, in order, each with the same five parts:
setup (checkpoint, thinking mode, GPU, data, split), inputs (what the method really has while
drafting against what an oracle supplied), metric (what was counted, over which rounds), timing
(what was timed, or that nothing was), and evidence (the file or job, the conclusion the numbers
support, and what they leave open). Completed results come first; proposed experiments are in
their own section at the end and are not evidence.

Three things that run through every entry:

- The split. Every saved answer is assigned by a hash of its prompt to train (60%), validation
  (20%) or test (20%) (`nexus/fit_gate.py`, `split_of`). Everything trained in November learned
  from training answers; every November number below is on validation answers, with their count
  stated; the test split has not been used.
- The two drafters. "Public" is `z-lab/Qwen3-4B-DFlash-b16`. "Draft-OPD" is
  `bingyang-lei/Qwen3-4B-Thinking-Draft-OPD`, a drafter post-trained for thinking-mode text.
  The target is always `Qwen/Qwen3-4B`, greedy, lossless.
- The November text. `specdec-data/long1`: answers the target wrote in thinking mode (reasoning
  on) to prompts from AIME (before 2024), MATH level 5, LongWriter and UltraChat, up to 16,384 new
  tokens, decoded one at a time; 366 training answers, 100 validation answers. `think1`: about
  8,000 more answers of the same kinds, up to 3,072 tokens, used only for training and for the
  count table. The class setting the project settled on is plain (non-thinking) mode; the
  November text is thinking-mode text, and no main-table number has yet been rerun on plain-mode
  text. This is an open item, not a result.

---

## Part A. September and October (public drafter, Modal A100, thinking off, GSM8K and UltraChat)

### A1. Train a better drafter (early September)

Setup: public drafter; target in plain mode; A100 on Modal; GSM8K and UltraChat; 64 dev prompts,
32 contexts each; no train/validation split beyond prompts held out from training.
Inputs: everything the drafter has in real drafting; no oracle.
Metric: a reward over fixed contexts (correct positions covered), not speed; original draft
retained in the best-of-two rows.
Timing: not measured.
Evidence: `results/SUMMARY.md`, Phase 1. Draft-OPD scored 13.9% above the base drafter on the
reward (real, not ours). Training on first rejection points: −0.5% to +0.2%, intervals cross zero.
A post-training pilot with verification credit: −6.7%. A full second drafter trained with
feedback: +10.21% kept, against +8.05% for a control without feedback; the feedback is worth
+2.16 points [1.17, 3.18]. Conclusion supported: retraining the drafter on our signal does not
transfer; a trained second drafter is worth about +10% positions. Open: none of this was timed.

### A2. Cheap substitutes and the first tree (September 24 audits)

Setup: as A1.
Inputs: no oracle.
Metric: positions covered on fixed contexts.
Timing: not measured.
Evidence: `results/SUMMARY.md`, Phase 1. Candidate mixing +0.2% on excluded prompts; cached
suffixes, cross-round logit transport (−25%), longer drafts, cheaper tree readout: all closed. A
24-node tree from the first drafter's own scores covered 565 positions against 443, for free, and
beat every substitute. Conclusion: the tree built from the first pass is the thing to build on.

### A3. Where the time goes (mid September; engineering)

Setup: public drafter; H100 (FA3) and A100; SGLang and vLLM.
Inputs: not applicable.
Metric: wall time of the target's check by number of rows.
Timing: measured directly.
Evidence: `experiments/cliff/`, `results/SUMMARY.md`, Phase 2. The check costs by tiles (32 rows
on H100 FA3, 128 on the GEMM); a draft crossing a tile boundary pays a whole tile; NVIDIA had
published the 128-row rule first. CUDA graphs for tree verification gave +25.1% on Modal; never
carried to the Nexus code, which runs eager. Deferred exact verification 1.30× at 89K context,
below the bar. Two results that anticipate Part B: feeding drafted prefix tokens back into the
drafter on Qwen3-8B hurt at its doubtful places (−22% and −35% of the gap), and target features
for that prefix recovered 78 to 84% but need a target pass; staging a block into 2 or 3
conditioned stages gave at most 1.04 to 1.05× tokens per round.

### A4. Conditional redrafting, the class project's first form (September 22 to 29)

Setup: as A1.
Inputs: the second pass reads the anchor, the first token guess, and masks; no oracle.
Metric: positions covered, original kept in best-of-two.
Timing: estimated from pass counts, not measured: even a free second pass would cut time per token
by at most 4.76% in this setting.
Evidence: `results/SUMMARY.md`, Phase 3. Untrained second pass alone −1.42%; best-of-two +5.13%;
trained second drafter with feedback +10.21%. Conclusion: a drafter re-reading its own draft
makes it worse; a separately trained one gives a useful alternative worth about +10%, less than
the pass costs.

### A5. Compress the second pass into a 3M-parameter editor (October 1 to 6)

Setup: public drafter; plain mode; A100; 384 training prompts, 1,152 updates per arm; dev prompts
held out.
Inputs: the editor reads the drafter's saved states; the teacher is a full second pass; no oracle
at use time.
Metric: positions covered on fixed contexts; tree coverage; original kept as fallback for the
teacher row.
Timing: not measured; the bar used was "a 5% round overhead".
Evidence: `results/SUMMARY.md`, Phase 4. Teacher +10.2%. Students +3.0% (correct-token loss),
+3.2% (prefix-weighted), +3.7% (teacher KL); candidate-focused distillation −1.5 points. Tree
coverage: original scores 14.2%, teacher 16.9%, best student 12.6%. The teacher learned its
prompts (repairs/damages 79/33 on training text, 35/41 on dev). A frozen DFlash fed its own
completed draft for ten passes dropped matching positions from 1,504 to 646 at pass 2. Late
October add-ons reached +8.5% (prefix), +10.0% (edit), +7.9% (control) in coverage. Public D2SD
weights on Qwen3-8B, one run of 8 prompts: +5.98% positions [−0.6, +12.7]. Conclusion: students
recover about a third of the teacher and lose to the free tree; the teacher does not transfer.
Open: nothing from Part A was measured on real generation. Two rules were adopted: fixed-context
coverage is not a speedup, and the baseline is the strong free tree, not the chain.

---

## Part B. November, replay studies of the second draft (both drafters, Nexus A5000, thinking on)

### B1 = Experiment 1. Why does a second draft help? The swap against the second run

Setup: public drafter (and Draft-OPD, and the public drafter with the October add-on weights
"off-a"); target in thinking mode; RTX A5000; the 100 validation answers of `long1`, replayed
(the rounds are plain decoding's own rounds; replay is exact for accepted tokens). Public drafter
181,141 rounds; Draft-OPD 120,835; off-a 71,361 rounds from 50 answers.
Inputs: every second draft here starts from the TRUE break, which is an oracle (in decoding the
break is not known); the token placed at the break is the drafter's own second choice, which is
available; the "right token" rows use the target's token, an oracle; the "target's real hidden
states" row is an oracle. Nothing else is supplied.
Metric: "got past" is the share of breaks at which the target accepts the token at the break;
"tokens added" is the mean extra accepted tokens per round with a break; the first draft is always
retained; all rounds with a break (99.3% of rounds on the public drafter, 97.0% on Draft-OPD).
Timing: none in this entry (see B4 and B5).
Evidence: `docs/WHY_A_SECOND_DRAFT_HELPS.md`, Results 1 to 5; jobs named there. Public drafter:
the swap (second choice at the break, old rest, no second run) gets past 36.1% of breaks and adds
0.646 tokens; the second run given the same token and rewriting the rest gets past 36.1% and adds
0.673 (+0.03 per break, range 0.017 to 0.036); the same drafter re-run freely on its own draft
gets past 12.5% and adds 0.245; the October add-on made to differ gets past 45.4%. Draft-OPD: the
swap 43.8% / 0.828, the rewritten rest 0.912 (+0.085), the right token is the second choice at
45.4% of breaks. Conclusion supported: a second draft helps by handing the target one more
candidate at the break; the second run itself adds about a tenth of that. Open: the share of the
second run's small gain that a trained pass could enlarge (B9 and B10 take this up).

### B2 = Experiment 2. What is a good ending worth? The target's hidden states

Setup: as B1.
Inputs: the right token at the break is an oracle in every row; the "old tokens" row uses what the
first draft already wrote; the "rest rewritten" row is a second run with the right token; the
"real hidden states" row is an oracle (the target's states for the tokens before the break, which
in decoding exist only after the next check).
Metric: extra accepted tokens after the right token, per round with a break.
Timing: none.
Evidence: `docs/WHY_A_SECOND_DRAFT_HELPS.md`, Result 4 (public drafter): old tokens 0.55, rest
rewritten by a second run 0.64, the drafter of the next round with the target's real hidden
states 2.03. Result 5 (Draft-OPD): with the right token, old rest 1.638, rest rewritten 1.826,
then the real hidden states 3.996; real states minus rewritten rest = 2.17 [1.94, 2.37]. For late
breaks (after 5 to 14 accepted tokens) rewriting the rest is worse than keeping the old rest
(−0.26 with the right token in place). Conclusion supported: inside the block the drafter has
tokens and not the target's states, and the states are what a good ending needs; showing it the
right token is not the same as giving it the state for that token. Open: whether a trained second
pass still needs the states (B10).

### B3. The restart at the break: does it need the states, and which (E1 of the restart study)

Setup: public drafter and Draft-OPD; thinking on; A5000; the 100 validation answers; 179,747
rounds with a break (public), 117,090 (Draft-OPD). Jobs `restart-pub-0`, `restart-opd-0`,
`nexus/restart_probe.py`.
Inputs: the drafter is restarted from the RIGHT token at the TRUE break (both oracles). What
varies is what it gets for the target's missing states of the tokens before the restart: the real
ones (oracle), nothing, the last real state repeated, older real states, only the nearest real
state, all but the nearest, or the real ones with noise.
Metric: tokens the target accepts after the right token when the drafter restarts there, per
round with a break.
Timing: none.
Evidence: `docs/SECOND_RUN_RESTART.md`, E1 table. Public / Draft-OPD: real states 2.17 / 3.65;
nothing 0.59 / 0.93; last state repeated 0.44 / 0.62; only the nearest real state 1.91 / 3.18;
all but the nearest 1.28 / 2.38; noise at 25% of each dimension's spread 2.14 / 3.62, at 100%
1.82 / 3.18, at 200% 1.11 / 2.10; for comparison the first draft's old tokens 0.55 / 0.64 and a
second pass inside the block 0.64 / 0.83. Conclusion supported: without the states a restart is no
better than the old tokens; one state, the nearest, carries most of it; a predictor must be within
about one spread of the real state to keep the gain. Open: see B6 (the predictor).

**Reconciling 2.17 and 3.65, and 2.17 again.** Three different quantities share these digits.
(a) B3's 2.17 is tokens accepted after the right token on the PUBLIC drafter when the drafter is
restarted at the break with the real states; B3's 3.65 is the same quantity on DRAFT-OPD. The two
differ because Draft-OPD is the better drafter (its plain chain gives 4.80 tokens a round against
3.20). (b) B2's 2.17 is a DIFFERENCE on Draft-OPD: real states (3.996) minus rewritten rest
(1.826), a measure of how much the states add over a token-only second run. (c) B2's 2.03 is the
public drafter's next-round value after the right token, which should match (a)'s 2.17 in
meaning; they come from two different mechanisms (B2 replays the next round's whole block, B3
restarts the drafter in place with the real states handed in) and agree within 0.14 on the public
drafter. On Draft-OPD the same pair is 3.996 − 1 = 3.00 (B2) against 3.65 (B3), a gap of 0.65 that
is NOT reconciled. The likely cause is that B2's rewritten rows are capped at the block's end while
B3's restart writes a full new block from the right token, but this has not been checked. Treat
B3's numbers as the restart's and B2's as the next round's; do not quote them as the same number.

### B4 = whole-answer replays of the second run against the swap (public drafter)

Setup: public drafter; thinking on; A5000; 50 validation answers replayed from start to end with
each method, the same 50 for every row; plain decoding 3.31 tokens per round.
Inputs: no oracle. Second drafts are placed at the four most likely breaks, scored from the
drafter's own confidence (the D2SD placement); the token at a break is the drafter's second
choice; "same drafter again" re-drafts freely with its first draft shown.
Metric: more accepted tokens per round over the first draft alone, all rounds, original
retained.
Timing: extra drafter time per round measured in replay (the second run's pass), not full rounds.
Evidence: `docs/WHY_A_SECOND_DRAFT_HELPS.md`, Result 6, second table. Same drafter again, one
redraft +5.3% (8.2 ms); same drafter again at four likely breaks +8.4% (12.8 ms); the swap at four
likely breaks, no second run, +18.8% (0.8 ms); second choice at four likely breaks, rest
rewritten by a second run, +20.2% (13.2 ms); the October add-on at four breaks +18.1% (13.0 ms),
made to differ +25.7% (13.4 ms). On the repaired drafter (off-a) the swap on top gives +52.3%.

**Reconciling +8.4% and +20.2%.** Both are whole-answer replays on the same 50 answers with
second drafts at the same four likely breaks. +8.4% is the second run LEFT FREE: shown its own
first draft, it usually repeats its first choice at the break, so it rarely gets past it. +20.2%
is the second run GIVEN the second-choice token at the break and rewriting the rest. The swap,
which gives the token and runs nothing, is +18.8%. So of the 20.2 points, 18.8 come from the token
and 1.4 from the run; the run costs 13.2 ms against the swap's 0.8 ms. This is the whole-answer
form of B1's per-break finding. Conclusion supported: the swap is the cheap, untrained method; a
second run on top of it is not worth its pass unless it gets much cheaper (B7, B8).

### B5 = the swap in real decoding, and what the larger check costs

Setup: public drafter and Draft-OPD; thinking on; A5000; real decoding of long validation
answers with every setting writing the same text (5 public, 7 Draft-OPD answers at the time of
the record); `nexus/time_check.py` for the check's cost.
Inputs: no oracle.
Metric: real tokens per second over plain decoding, all rounds.
Timing: full rounds; the check by amount of text: first draft alone 48.5 ms at 1,000 to 4,000
tokens of text, 54.8 at 8,000, 73.9 at 12,000, 97.3 at 16,000, 135.9 at 24,000; with four
alternatives of 4 tokens +0 to +3.5 ms; the mask costs 0.4 ms.
Evidence: `docs/WHY_A_SECOND_DRAFT_HELPS.md`, Result 8; `docs/DIRECTION.md`, item 4. The swap
gives +13% real tokens per second on the public drafter and +10% on Draft-OPD. Side finding: in
16-bit arithmetic the tree-style check rounds differently from plain decoding and on about 0.6% of
tokens a near tie flips, so no tree check matches plain decoding bit for bit. Conclusion
supported: the swap is a real speedup with no training. Open: small answer counts.

### B6. A predictor of the missing states (E3 of the restart study)

Setup: Draft-OPD and public; thinking on; A5000; trained on training answers at real breaks;
measured on validation answers (12, 3,335 rounds for the restart rows).
Inputs: the predictor reads only what exists at the time of the second run (the drafter's own
final states at the token and the one after it, the token, the last real vector before the
round); its targets at training time are the real vectors (oracle in training only).
Metric: tokens right of the three checked after the restart's first token; share of the gap
between "nothing" and "real states" closed.
Timing: none.
Evidence: `docs/SECOND_RUN_RESTART.md`, "the second predictor's result"; `docs/DIRECTION.md`,
November 18 night. Predicted states close 28% of the gap on Draft-OPD (2.07 against 1.11 with
nothing and 4.51 with real states) and 49% on the public drafter (1.40 against 0.59 and 2.24); the
frozen whole drafter gets 1.37 right of 3 with predicted states, 2.09 with real, 0.75 with none.
Conclusion supported: a predictor recovers a third to a half; the states cannot be predicted well
enough from what the drafter has. This is the "hidden states" lever and it did not pay.

### B7 = Experiment 3. The second run in real decoding, and DDTree as the new baseline (Draft-OPD)

Checkpoint change: Experiments 1 and 2 were run on the public drafter first (with Draft-OPD
replications); from this experiment on, every tree and second-drafter number is on Draft-OPD only.
The public drafter's rows stop here.

Setup: Draft-OPD; thinking on; A5000; 16 long validation answers, the same text in every row
(`probe-shift-opd-0`).
Inputs: no oracle; restarts at the four likely breaks from the next choice and one token earlier,
with predicted states (B6's predictor), three tokens checked after each; DDTree with 63 nodes from
the first run's scores.
Metric: tokens per round and real tokens per second, all rounds, original retained.
Timing: draft, second run, check and the whole round, measured.
Evidence: `docs/DIRECTION.md`, November 18 afternoon. Plain 4.82 tokens a round (draft 10.2 ms,
check 57.5) at 70.1 tokens per second; the second run 6.24 (+30%) with 14.7 ms of second run, 73.2
tokens per second (+4%); DDTree 6.59 (+37%) for 1.6 ms, 88.1 tokens per second (+26%); DDTree with
the second run on top 7.14 (+48%) for 17.3 ms, 78.2 tokens per second (+12%), slower than DDTree
alone. ReAnchor gave its published +15% tokens but −1% speed (its blocks lengthen the first pass
by 5.7 ms). Conclusion supported: a whole second pass does not pay on this target in this stack;
DDTree is the baseline from here on. Derived, not measured: at 4 ms the second run would be 2%
faster than DDTree with today's tokens.

### B8. Learning during the answer and the long-text window (both drafters)

Setup: public drafter mainly, Draft-OPD for the decisive rows; thinking on; A5000; validation
answers up to 16,384 tokens; real decoding for Result 5.
Inputs: the drafter's extra weights update from the target's own choices during the answer; no
oracle (the signal is what the target already wrote).
Metric: real tokens per second; replay tokens per round for the cheaper variants.
Timing: full rounds in Result 5; the cost of a learning step by text length in Result 4.
Evidence: `docs/LEARNING_DURING_THE_ANSWER.md`, Results 1 to 9 and "Where this leaves the main
line". Showing the public drafter only the last 3,072 tokens: +25% real tokens per second (+61%
beyond 8,000 tokens); add-on weights taught ahead of time: +42%; with the swap on top +42% and
+59%. Learning during the answer loses in real time to weights taught once from as few as ten
answers, on both drafters, and on Draft-OPD it loses to doing nothing; what it learned was a
repair of attention for the current length. Conclusion supported: closed, except for lengths
beyond anything taught ahead of time. Open: the 32,768-token replays named under "Still to come"
were not completed before the line was dropped.

---

## Part C. November 18 to 27, the second run made cheap and then replaced (Draft-OPD only)

### C1 = Experiment 4. The one-layer second drafter, restarting at the break

Setup: Draft-OPD; thinking on; A5000; trained on 366 training answers (one pass; a second pass
over the same answers overfits), measured on 12 validation answers (3,335 rounds); the every-node
tree on 8 validation answers (7,915 rounds).
Inputs: one trained layer of the drafter restarts at the alternative token; it reads the tokens
(and in one variant the states written into the block); no oracle at use time; "real states"
columns are oracles for comparison.
Metric: of the three tokens checked after the restart's first token, how many are right; tokens
per round in replay for the tree.
Timing: a pass costs by its layers, about 1.7 ms each, and hardly by the restarts it holds
(measured); the one-layer pass 6.9 ms in later real-speed runs.
Evidence: `docs/DIRECTION.md`, November 18 night; `docs/SECOND_DRAFTER_RESEARCH.md`. One layer
reading the tokens: 0.87 right of 3 with predicted states, 0.90 with real, 0.86 with none (the
frozen whole drafter: 1.37 / 2.09 / 0.75). The layer learned evenly through its pass (0.65 to 0.95
by fifths) and was still rising when the answers ran out; a second pass over the same answers
reached 1.23 on them and 0.92 on validation, so it learned those answers. Conclusion supported: a
trained layer gets most of a whole drafter's restart quality without the states, and is limited by
how little text it saw, not by its size; the "second run" of Experiments 1 to 3 can be a layer.

### C2 = Experiment 5. The count table (DForkTree's scores, written from the paper)

Setup: Draft-OPD; thinking on; A5000; counts from the training answers of `long1` and all of
`think1`; measured on 8 validation answers (replay) and 3 (real speed).
Inputs: the first run's candidates and a lookup of which token the target wrote after which two
tokens in the training text; nothing trained; no oracle.
Metric: tokens per round in replay (own rounds); real speed over plain, all rounds.
Timing: the table lookup 2.3 to 3.1 ms a round; full rounds in the speed runs.
Evidence: `docs/SECOND_DRAFTER_RESEARCH.md`, November 19 sections; `docs/DIRECTION.md`, "The
second run on top of DForkTree's counts". Right after a fork 39.5% (first run 20.7%). Replay 8.31
tokens a round against DDTree's 7.56. Real speed +28 to +30% over plain against DDTree's +21%.
The one-layer second drafter on top of the counts: +3.4% tokens for 8 ms, not faster than the
counts alone. Conclusion supported: the count table is the strongest free method and the bar any
learned correction must beat.

### C3 = Experiment 6. The pair scores, a corrector that needs no pass

Setup: Draft-OPD; thinking on; A5000; trained on training answers of `long1` plus `think1`
(0.6 hours for `opd-p1`, about 1,500 answers; 2 hours, 2 passes for `opd-p2` and `opd-p2s`);
12 validation answers for fork accuracy, 8 (replay) and 3 (speed).
Inputs: for each node, the drafter's state at the place, the input embeddings of the two tokens
before (from the branch), the candidate's output-head row, and the first run's log-probabilities of
the candidates; `opd-p2s` additionally the three numbers about the token before the place (the
first run's log-probability of it, whether it was the first choice, the first choice's
log-probability), all computed in the first pass; no oracle.
Metric: accuracy right after a fork; tokens per round in replay; real speed, all rounds.
Timing: the scores' tables and search 4.5 ms a round (eager, numpy search); full rounds.
Evidence: `docs/SECOND_DRAFTER_RESEARCH.md`, "Pair scores: result" and "the first clear win";
`nexus/results/next2-replay`, `next3-speed`, `diag5-rows`. Fork accuracy: `opd-p1` 40.2%,
`opd-p2` 42.4%, `opd-p2s` 47.5%. Replay (7 answers): `opd-p2` 8.19, `opd-p2s` 8.67. Real speed (3
answers, same text): plain 5.80 tokens a round at 64.0 ms; DDTree 7.56 at 68.7 ms, +21.2%;
`opd-p1` 8.26 at 72.5 ms, +25.6%; `opd-p2s` 8.75 at 73.2 ms, +31.8%. Conclusion supported: told
how sure the first run was, the pair scores are the first method of ours clearly past DDTree in
real speed, past the count table, with no pass and no stored text. Open: 3 answers; eager timing;
whether the three numbers do more than a yes/no flag (proposed P1).

### C4 = Experiment 7. The diagnosis: why the answer's path leaves each tree

Setup: Draft-OPD; thinking on; A5000; 7 validation answers, 6,993 rounds of DDTree; every tree
built on the same rounds (`nexus/trace_trees.py`, job `next1-trace`).
Inputs: the saved answer gives the oracle of which path is right, used only to classify why each
tree's path stopped; the trees themselves use nothing extra.
Metric: share of rounds stopped by each cause (block end; right token not among the 32
candidates; ranked first but the node budget went elsewhere; ranked 2nd to 4th; ranked 5th or
lower); tokens per round; rounds further or less far than DDTree; "following the answer" tokens
when only the best b children are kept at every node.
Timing: none.
Evidence: `nexus/results/next1-trace/trace.json`; `docs/SECOND_DRAFTER_RESEARCH.md`, "The
diagnosis" and the November 27 reading. DDTree: block end 7.2%, not a candidate 1.1%, ranked first
not kept 8.6%, ranked 2 to 4 42.7%, ranked 5 or lower 40.4%. `opd-p2` against `opd-p2s`: 9.9% →
11.9%, 9.9% → 12.1%, 13.5% → 12.5%, 38.4% → 38.6%, 28.3% → 24.9%. Against DDTree on the same
rounds: `opd-p2` further in 41.9% of rounds, less far in 10.3% (gaining 0.87, losing 0.18 tokens a
round); `opd-p2s` further in 49.4%, less far in 4.5% (gaining 1.16, losing 0.06). Keeping only the
first choice at every node: DDTree 6.00, `opd-p2` 6.03, `opd-p2s` 6.39. The four-layer second
drafter (C5) cuts the two ranking causes to 10.8% and 10.2%. Conclusion supported: the three
numbers act about 70% through better ranking and 30% through less damage to DDTree's good
rounds, consistent with one mechanism (the strength of the correction); the budget cause (12.5%,
0.28 tokens a round if fixed) is untouched by every corrector here. Open: whether a yes/no flag
gives the same (P1).

### C5 = Experiment 8. Deeper second drafters, and every-node correction

Setup: Draft-OPD; thinking on; A5000; two and four of the drafter's layers trained as C1
(`opd-e2`, `opd-e4`); 12 / 7 / 3 validation answers as in C3.
Inputs: the second pass reads the branch's tokens and the first run's states; no oracle.
Metric: fork accuracy; tokens per round (one pass of rows for the first tree's nodes, and "every
node" with more passes until every node has a row); real speed.
Timing: the two-layer pass about 16 ms, four-layer about 27 ms, eager.
Evidence: `nexus/results/next1-trace`, `next2-replay`, `next3-speed`. Fork accuracy: one layer
49.1%, two 55.6%, four 57.3%. Replay (7 answers): one layer 8.21, two 8.35 (8.71 every node), four
8.48 (8.94 every node). Real speed: one layer 8.29 tokens a round at 75.3 ms, +21.3%; two 8.51 at
84.7 ms, +10.7%; four 8.63 at 95.7 ms, −0.6%. Conclusion supported: deeper second drafters rank far
better and lose on time; the cheaper corrector wins in speed. Their every-node numbers are the
practical ceiling for correctors of this family (8.94 against 8.67).

### C8. The ablation of the three numbers: the flag alone does it (November 27, 07:00 UTC)

Setup: Draft-OPD; thinking on; A5000; each variant trained exactly as `opd-p2s` (2 passes, 2
hours, about 5,260 answers of `long1` train plus `think1`); fork accuracy on 12 validation answers;
replay on the same 8 validation answers (6,870 to 6,900 rounds). Jobs `abl1-top1`, `abl2-logp`,
`abl3-scalar`, `abl4-seed`.
Inputs: as C3, with the three numbers cut down: only "was the token before the first choice"
(`opd-p2s-t`); that and the token's log-probability (`opd-p2s-tl`); all three but one scale per
token instead of per rank (`opd-p2s-g`); all three again with seed 1 (`opd-p2s-s1`). No oracle.
Metric: accuracy right after a fork (true prefix); tokens per round in replay, own rounds.
Timing: none (the cost is the same for all).
Evidence: `nexus/results/abl1-top1` to `abl4-seed`, `status.json` and `replay/status.json`.
Right after a fork: `opd-p2s` 47.5%, flag only 47.2%, flag and log-probability 47.3%, one scale
per token 47.9%, seed 1 47.0%; without any of them (`opd-p2`) 42.4%. Tokens a round: flag only
8.674, flag and log-probability 8.681, one scale per token 8.713, seed 1 8.695, against `opd-p2s`
8.667 (7 answers) and `opd-p2` 8.19 (7 answers). The spread across all four is 0.04 tokens a
round and the seed-to-seed difference 0.03. Conclusion supported, by the rule fixed in advance
(flag alone at 8.5 or more): the gain of the three numbers is entirely the yes/no flag "was the
token before the first run's own first choice", and a single learned scale per token carries it
as well as a scale per rank. The continuous confidence adds nothing measurable. What stands is
narrower than the November 27 reading claimed: a learned per-node corrector needs DForkTree's
on-path/off-path distinction as an input, and with it gains +0.48 tokens a round (+5.9%) over the
same corrector without it; DForkTree's fixed weights are the same idea in a count table. This is
not new in kind. Open: whether the flag's gain is the same for other correctors (TreeFlash's head,
GRAFT's), which would make it a general finding about learned correctors rather than ours.

### C9. Can the drafter's alternatives be read out early? (November 27, 07:05 UTC, untrained)

Setup: Draft-OPD (5 layers); thinking on; A5000; 24 validation answers of `long1`, an anchor every
16 tokens (10,220 first runs); the states after every layer read with the drafter's final norm and
the output head, untrained (`nexus/early_readout.py`, job `probe1-readout`, 1 minute).
Inputs: the readouts use only the first run's own states; the answer is used only to score them.
The four breaks are chosen by each readout's own confidences with the break-score rule (and, for
comparison, by the last layer's confidences with the early readout's tokens).
Metric: the right token in the readout's top 1 / 2 / 8 over the block's places 1 to 15; the
first choices right in a row from the block's start; of the four branches (first choices before
the break, second choice at it), the share whose whole prefix matches the answer.
Timing: the readout (norm and head over a block of 16) 1.18 ms, measured.
Evidence: `nexus/results/probe1-readout/status.json`. Top 1 / top 2 / right in a row: after layer
1: 3.7% / 5.6% / 0.22; layer 2: 10.1% / 14.9% / 0.66; layer 3: 21.6% / 30.1% / 1.51; layer 4: 33.0%
/ 42.1% / 2.73; last layer: 44.7% / 53.5% / 5.28. Branches with the whole prefix right, own
breaks: layer 2 4.7%, layer 3 7.0%, layer 4 7.3%, last layer 9.8%; with the last layer's breaks and
the early tokens: 2.2%, 4.0%, 6.1%. Conclusion supported: untrained, the early readouts are poor
(at layer 3 the chain of first choices is right for 1.5 tokens against 5.3 at the end), so the
untrained version of branching inside the drafter has no useful inputs; the whole-prefix share
falls less (7.0% against 9.8%) only because the poorer confidences place the breaks earlier
(mean place 2.6 against 4.3), where short prefixes are easier. Note also the ceiling this metric
puts on the four-branch structure itself: even the final readout gives 9.8% of branches a right
whole prefix, about 0.4 useful branches a round. Open: whether a trained intermediate readout
(LayerSkip-style) closes the gap, which is the next test if the cost probe (P10) is affordable.

### C10. Real speed on the standard benchmark prompts, thinking off (November 27, 07:40 UTC)

Setup: Draft-OPD; target with thinking OFF (the class setting); A5000; 10 prompts each of GSM8K,
MATH-500, HumanEval and MT-Bench (the first turn, `dflash/benchmark.py`'s fixed selection), the
target's own answers, every setting writing the same text; counts from `long1` train and `think1`
(job `conf3-bench`, 17 minutes; 3,095 to 4,355 rounds).
Inputs: no oracle. Note that the drafter and the corrector were trained on thinking-mode text and
are used here on non-thinking answers.
Metric: tokens per round and real tokens per second over plain, all rounds.
Timing: full rounds (eager).
Evidence: `nexus/results/conf3-bench/status.json`. Over all 40 prompts: plain 7.17 tokens a round
at 64.6 ms; DDTree 8.89 at 69.6 ms, +15.1%; count table 9.64 at 72.0 ms, +20.6%; pair scores
with the flag (`opd-p2s`) 10.09 at 73.4 ms, +23.8%. By benchmark (DDTree / count table / pair
scores over plain): GSM8K +25.6 / +29.4 / +33.6%; MATH-500 +3.9 / +12.6 / +15.1%; HumanEval
+32.9 / +28.6 / +32.5%; MT-Bench +35.7 / +38.4 / +43.2%. Conclusion supported: on the standard
prompts with thinking off, the corrector is +7.6% over DDTree and +2.7% over the count table in
real speed (eager), the same picture as on the long thinking answers (+8.7% over DDTree on 3
answers); it loses nothing on code (HumanEval ties DDTree where the count table falls behind).
Its extra cost is 3.8 ms a round. Open: 10 prompts a benchmark; graphed timing (`cost2-search`).

### C6. The published second passes made here (D2SD, D-Loop)

Setup: Draft-OPD; thinking on; A5000; `nexus/train_loop.py` (D-Loop: the same drafter retrained
with prefix and suffix views, β = 0.7, τ = γ = 7, learning rate 2e-4; D2SD: a separate drafter
trained the same way, β = 0.85, rate 1e-4, all weights).
Status: no completed result. The first D2SD training (`loop1-d2sd`) ran 0.5 hours (1,048 answers,
loss falling) and died at its first checkpoint save when the home folder filled; the first
D-Loop training (`loop2-dloop`) died out of memory and was requeued with smaller groups. Both are
queued again (`loop3-dloop`, `loop4-d2sd`). Nothing from this entry is evidence yet.

### C7. The search done on the GPU, and as a CUDA graph (engineering, queued)

Setup: as C3; `CountTree(fast=True)` searches in tensors of fixed shape, `graph=True` records the
whole build as one CUDA graph; the tree is the numpy search's (test
`test_tree_with_counts_searched_in_tensors`).
Status: queued as `cost1-search`; no measurement yet. The purpose is the corrector's 4.5 ms,
against GRAFT's 0.3 ms, DForkTree's 0.9 ms tree build, and DSpark's head at 0.2 to 1.3% of a
round.

---

## Part D. What the three literature readings established (November 26 to 27)

Known before us and confirmed by our measurements: the hidden-state cause (DForkTree 2611.12356:
top-1 after a fork 29% → 84% with the target's features, 37 to 39% with tokens); a second pass not
paying for itself (DForkTree's LoRA pass +1 to 2.9% tokens for 2.5 to 6.2 ms); one or two tokens
carrying most of the branch information (2608.27339; DForkTree's bigram and trigram counts, no
4-gram gain); distilling a re-anchored drafter into a head (DForkTree's appendix, never compared
with target-label training); over-strong corrections hurting (GRAFT's β sweep); drafter passes
nearly free in positions (MFD 2611.13017: 3.70 ms at 16 positions, 4.24 at 64).

The published comparator for C3: TreeFlash (2606.03819, EMNLP 2026 Findings), a one-token head
after the backbone, +12.4% tokens and +9.1% speed over DDTree at 64 nodes on Qwen3-4B. Ours is
+14% tokens (7 answers) and +8.7% speed (3 answers) over DDTree: parity, at 4.5 ms where TreeFlash's
head is cheaper. Published margins over DDTree with the same drafter: DForkTree +15.1%, TreeSpark
+9 to +16% tokens, Weaver +6 to +15%, UBTree +25.3% with a retrained drafter.

Ours, as far as the readings found: splitting the second draft's gain into token, run and states
on the same rounds (B1, B2); the per-cause accounting of why a tree's path stops with each
method's gain attributed to causes (C4); continuous first-run confidence about the token before a
place as an input to a learned per-node scorer, with its measured gain and its trace reading (C3,
C4). C8 settled the last: it is not distinct from DForkTree's on-path flag; the flag alone gives
the whole gain.

Withdrawn on November 27: drafting the tree in the drafter's first pass by giving extra positions
the alternative token at a fork. The alternative token is an output of that pass and cannot be its
input; any repair is a second pass, which is C1 and C5.

---

## Part E. Proposed experiments (not evidence; each with what it must show)

P1, now done (November 27, 07:00 UTC; moved to C8 below). The result: the flag alone does the
whole job.

P2. Confirmation of C3's speed (queued, `conf1-speed`: up to 30 long validation answers; DDTree,
count table, `opd-p2`, `opd-p2s`) and on 40 standard-benchmark prompts with thinking off
(`conf2-bench`: GSM8K, MATH-500, HumanEval, MT-Bench; `live_speed.py --bench`). Rule: a margin
over DDTree below +5% ends the speed claim.

P3. The corrector's cost (queued, `cost1-search`): numpy search against the tensor search against
the CUDA graph, same trees, 4 answers. Target under 1.5 ms. Derived: at 0.8 ms and the same
tokens the margin over DDTree would be about +14%.

P4. The retrained drafters (queued, `loop3-dloop`, `loop4-d2sd`), then a probe with nothing,
predicted and real states (B3's question for a trained pass), and replay with DDTree on the same
answers. Rule: if a retrained drafter with plain DDTree is close to or above `opd-p2s`, the main
line after the class moves to the drafter.

P5. The budget cause (C4's 12.5%, 0.28 tokens a round): a few learned temperatures over depth and
on/off the path on top of `opd-p2s`, fit on training answers; one job.

P6. The coverage cause (C4's 12.1%): 4 to 8 trigram-proposed candidates at nodes after a fork
(DForkTree reports they hold 37 to 41% of the missing tokens); one job.

P7. The ranking cause: the four-layer drafter as a teacher for the pair scores at mixing weights
0, 0.25, 0.5, 1 against target labels alone, equal data (not published anywhere); and the pair
scores told the first run's probability of the whole branch prefix and the distance from the fork
rather than two tokens; two jobs.

P8. A 127-node tree, since the check is flat to the tile (A3); one job.

P10. Branching inside the drafter (proposed November 27). Two checks before any training: C9
(done: untrained early readouts are poor) and the cost probe `probe2-cost` (queued:
`nexus/branch_cost.py`, every stage timed on real rounds at 1,024 / 4,096 / 8,192 tokens of
text, the check by rows, and the token gain each setting needs to break even against DDTree). The
decision table: useful inputs and affordable cost, train the branching drafter; poor inputs and
affordable cost, a small early-readout training test first; too expensive, stop. C9 says the inputs
are poor untrained, so the only path forward is the readout-training test, and only if the cost is
affordable.

P9. Plain-mode text: generate answers with reasoning off and rerun the main table, since the class
setting is plain mode and all November numbers are on thinking-mode text.

Bars, fixed now: a workshop result needs graphed real speed of +12 to +15% over DDTree on the
standard suite, +3 to +5% over the count table on the same drafter, the corrector under 1 ms, and
P1 showing where the gain comes from. Below that it is a class report of a measured chain.

---

## Dates, constraints, and where things are

The class report is due December 3. Copy everything off the cluster by December 28; the account is
deleted January 1, 2027. Home folder quota 32 GB (an unused 6.6 GB environment was removed on
November 27 after it filled the folder and silenced the runner for three hours). The runner
restarts with one line on nexusclass01; jobs delete nothing; job code copies live under
`/tmp/specdec-code-$USER/`.

Records: `results/SUMMARY.md` (Part A, written November 3); `docs/WHY_A_SECOND_DRAFT_HELPS.md`
(B1, B2, B4, B5); `docs/SECOND_RUN_RESTART.md` (B3, B6); `docs/LEARNING_DURING_THE_ANSWER.md`
(B8); `docs/DIRECTION.md` (B7, C1, C2, day by day); `docs/SECOND_DRAFTER_RESEARCH.md` (C1 to C5,
Part D); `nexus/results/<job>/` (every job's status, answers, traces); `nexus/queue/<job>.sbatch`
(what each job ran and why). Checkpoints on the cluster in `~/specdec-ckpt`: `warm-opd-e1/e2/e4/
e1m1.pt`, `pairs-opd-p1/p2/p2s.pt`, `restart-opd-m0.pt`, and the drafters being trained.
