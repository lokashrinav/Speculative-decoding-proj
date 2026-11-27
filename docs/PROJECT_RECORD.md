# The project in one file: what was asked, what was measured, what it means

Written November 27, 2026. This gathers the whole project, September to now, in plain words.
Every number says which experiment and which drafter it comes from, and whether it was
measured, derived, estimated, or taken from a paper. The detailed records it draws on are
listed at the end.

## The setting

The target model is Qwen/Qwen3-4B, decoded greedily, so every method must produce exactly the
target's own text (lossless). The drafter is a DFlash drafter: a small model that writes a block
of 16 tokens in one parallel pass, reading the target's hidden states of the text so far. The
target then checks the block in one pass and keeps the tokens up to its first disagreement, plus
its own token there. Two drafters were used. The public drafter (`z-lab/Qwen3-4B-DFlash-b16`) was
used in September, October, and the first November experiments. Draft-OPD
(`bingyang-lei/Qwen3-4B-Thinking-Draft-OPD`), a post-trained drafter made for thinking-mode text,
was used for everything about trees and second drafters from mid-November on.

The text used in November is long answers written by the target itself (folder `long1` for
training and validation, `think1` for more training text). September and October used GSM8K and
UltraChat prompts on an A100 (Modal, about $100 in total). November runs on the UMIACS Nexus class
cluster (RTX A5000, 24 GB, up to four at once), driven by a job runner that polls a git branch.

A round on the A5000 in our eager PyTorch code costs about 64 ms: the drafter's pass about 9 to 10
ms, the target's check 55 to 64 ms, and the check is nearly flat in the number of tree nodes up to
a tile of 128 rows. Published serving stacks do the same round in 12 to 19 ms with CUDA graphs.
That difference matters for every speed number below: a method that costs a few milliseconds is
cheap here and expensive there.

## The question the project started from

A drafter guesses a block; the target checks it. Can a second look at the draft, by the drafter
or by something smaller, make the draft better, cheaply enough to speed decoding up?

## Phase 1, early September (public drafter, Modal, measured in coverage, not speed)

Training a better drafter was tried first. A published post-trained drafter (Draft-OPD) scored
13.9% higher than the base drafter on our reward, which is real but not ours. Training on first
rejection points gave no gain that transferred (−0.5% to +0.2%). A post-training pilot with
verification credit lost 6.7%. Both closed.

Keeping both the first draft and a second draft and letting the target pick gave +5.13%. A full
second drafter trained with feedback kept +10.21%, against +8.05% for a matched control without
feedback, so the feedback itself was worth +2.16 points, concentrated at draft places 2 to 4. An
adapter with two private layers that reused the draft's states gave +4.90% at a cost per accepted
position 8.57% worse than plain drafting. The arithmetic said that even a free second pass would
cut time per token by at most 4.76% in that setting.

Audits on September 24 closed the cheap substitutes: candidate mixing (+0.2%), cached suffixes,
cross-round logit transport (−25%), longer drafts, cheaper tree readout. A 24-node tree built from
the first drafter's own scores covered 565 positions against 443 for free and beat all of them.
This was the first sign that the tree, not the second pass, was the thing to build on.

## Phase 2, mid September: where the time goes (engineering, not novel)

The target's check costs by tiles: flash attention verifies rows in tiles (32 on H100 FA3, 128 on
the GEMM), so a draft that crosses a tile boundary pays for a whole extra tile. We wrote length
caps for SGLang and vLLM, then found NVIDIA had published the 128-row rule first. CUDA graphs for
tree verification (TreeFlash in our code) gave +25.1% on our Modal setup; this was never carried
to the Nexus code, which still runs eager. Deferred exact verification reached 1.30× at 89K
context, below the bar. Feeding drafted prefix tokens back into the drafter on Qwen3-8B hurt at
the drafter's doubtful places (−22% and −35% of the gap); target features for that prefix recovered
78 to 84% but need a target pass. Staging a block into 2 or 3 conditioned stages gave at most
1.04 to 1.05× tokens per round. These two last results anticipate the November finding about
hidden states.

## Phase 3, September 22 to 29: conditional redrafting (the class project's first form)

A second drafter pass reads the anchor, the first token guess, and masks. Untrained, the second
pass alone was −1.42% against the original draft. Keeping both drafts and letting the target pick
gave +5.13%; a trained second drafter with feedback +10.21%. The conclusion of this phase: a
drafter re-reading its own draft makes it worse; a separately trained second drafter produces a
useful alternative on some contexts, worth about 10% more correct positions, which is less than
the second pass costs.

## Phase 4, October 1 to 6: compress the second pass into a small editor (public drafter)

A 3-million-parameter editor read the drafter's saved states and was trained against a full
second-pass teacher (384 training prompts, 1,152 updates per arm). The teacher, with the original
kept as fallback, gave +10.2%. The students, with three losses, gave +3.0% (correct token), +3.2%
(prefix weighted), +3.7% (teacher KL). None beat a 5% round overhead, and candidate-focused
distillation was worse than the untouched draft (−1.5 points). In tree coverage the original
scores reached 14.2%, the full teacher 16.9%, the best student 12.6%. The reason: the teacher
learned its training prompts (repairs/damages 79/33 on training text, 35/41 on dev text), and
reading the whole completed draft was worse still (25/94). A frozen DFlash fed its own completed
draft for ten passes dropped matching positions from 1,504 to 646 at the second pass and stayed
there. The public D2SD weights on Qwen3-8B, one run of 8 prompts, gave +5.98% positions with four
conditional branches (interval −0.6 to +12.7).

Late October add-ons (students of the second pass with different inputs) reached +8.5% (prefix),
+10.0% (edit), +7.9% (control) in coverage, again on fixed contexts rather than real generation.

The verdict written on November 3: the second pass is worth having and too expensive to run; that
gap is real and is the open question; our students had not closed it and our teacher did not
transfer. Two rules were written down after this phase and are still in force: fixed-context
coverage is not a speedup, nothing counts until measured on real generation; and compare against
the strong free tree, not against the plain chain.

## Phase 5, November 10 to 17: why a second draft helps at all (both drafters, replay)

This was the first experiment on the cluster, over 181,141 rounds of 100 answers on the public
drafter and 120,835 rounds on Draft-OPD, with every kind of second draft built on the same rounds.

The finding: a second draft helps because it hands the target one more candidate token at the
place where the first draft breaks, and almost nothing else. The control that proves it is the
swap: the first draft with the drafter's own second-choice token at the break and nothing else
changed. The swap gets past 36.1% of all breaks. A real second run given the same token gets past
the same breaks and adds only 0.03 tokens per break on the public drafter (0.09 on Draft-OPD).

The second finding: what is missing after the break is the target's hidden states. With the right
token placed at the break, the first draft's old tokens give 0.55 more accepted tokens, a second
run that rewrites them inside the block gives 0.64, and the drafter in the next round, which by
then has the target's hidden states, gives 2.03. A second run costs about 13 ms on a round of about
62 ms; in replay it gave +20.2% tokens against +18.8% for the swap, which is no gain in speed. In
real decoding the swap alone gave 13% more tokens per second on the public drafter and 10% on
Draft-OPD, with no training and no second run.

A side finding: when the target checks a tree of drafts in one pass, its 16-bit arithmetic rounds
differently than in plain decoding, and on about 0.6% of tokens a near tie flips. No tree-style
check can promise plain decoding's text bit for bit in 16 bits.

We found no paper that splits the gain of a second draft, on the same rounds, into the token at
the break, the second run, and the target's hidden states. DForkTree (2611.12356) later confirmed
the hidden-state cause independently: with the target's features at a fork the drafter's top-1
accuracy goes from 29% to 84%; with tokens only, 37 to 39%.

## Phase 6, November 16 to 17: learning during the answer (public drafter and Draft-OPD)

The public drafter falls off on long answers because it sees more text than it was trained on.
Showing it only the last 3,072 tokens gave 25% more real tokens per second (61% beyond 8,000
tokens); small add-on weights taught ahead of time gave 42%. Learning during the answer, which
updates the drafter's extra weights as the answer is written, lost in real time to weights taught
once from as few as ten answers, on both drafters, and on Draft-OPD it lost to doing nothing. What
the drafter learned during an answer was mostly a repair of its attention for the current text
length. Closed, except for answers beyond the lengths anything taught ahead of time has seen.

## Phase 7, November 17 to 18: the second run restarted at the break, and what to feed it (Draft-OPD)

If the token at the break is the easy part and the tokens after it are the hard part, the second
run should restart the drafter at the alternative token as the next round would, and the question
is what to give it in place of the target's missing hidden states. The limit was measured first: a
restart at the four likely breaks with the target's real states gives +34.5% tokens per round on
the public drafter and +40.1% on Draft-OPD over the first draft alone (the swap gives +17.9% and
+14.8%). At a 13 ms second run that limit is worth about +11% and +17% in real speed.

Stand-ins for the states were tried: none, the drafter's own, and a trained predictor. The
predictor was the worst-returning piece of the project: of the three tokens checked after the
restart's first token, the frozen whole drafter got 1.37 right with predicted states, 2.09 with
real states, 0.75 with none (12 validation answers, 3,335 rounds). Real decoding on 16 long
answers (Draft-OPD, the same text in every row): plain 4.82 tokens per round at 70.1 tokens per
second; the second run at four breaks 6.24 tokens (+30%) at 73.2 tokens per second (+4%), because
it cost 14.7 ms; DDTree with 63 nodes 6.59 tokens (+37%) at 88.1 tokens per second (+26%) for 1.6
ms; DDTree with the second run on top 7.14 tokens (+48%) at 78.2 tokens per second (+12%), slower
than DDTree alone. The conclusion: a whole second pass does not pay on this target; it must get
cheap, and DDTree is the baseline from here on.

ReAnchor (published) gave its +15% tokens per round in our decoder but made the first pass 5.7 ms
longer, −1% in speed as published. A table of what the target already said added 6 points to
DDTree on answers that repeat themselves and little elsewhere.

## Phase 8, November 18 to 19: the cheap second run (one layer) and the count table (Draft-OPD)

The second run was cut to one trained layer of the drafter, restarting at the break. Trained at
every place with the true tokens before it, on candidates only, a one-layer second drafter is right
after a fork in 49 to 51% of the places where the first run is right in 21% and DForkTree's count
table in 39.5% (12 validation answers). About 1,000 answers are enough for that; 300 give most of
it; a layer from new weights gets two thirds of it, so the training and not the reuse of the
drafter's layer is what makes little text enough. A whole second pass costs by its layers, about
1.7 ms each, and hardly by how many restarts it holds.

DForkTree's count table (which token the target writes after which two, from the training text,
no training, one lookup) was written from the paper and measured: 8.31 tokens per round on the 8
validation answers against DDTree's 7.56, and +28 to +30% real speed over plain against DDTree's
+21%. The one-layer second drafter alone reached 8.21 tokens and +21.6% speed, because its pass
costs 6.9 ms. Every learned correction on top of the counts was measured in real speed, and none
was faster than the counts alone.

## Phase 9, November 19: the pair scores, and the diagnosis (Draft-OPD)

The knowledge of the one-layer second drafter was moved into something that needs no pass: a
small network that scores a candidate token given the two tokens before it, the drafter's state at
the place, and the candidate's output-head row (the form of the DFlash 2 and UBTree selectors).
Trained 0.6 hours (`opd-p1`), it reached 40.2% right after a fork, 8.09 tokens per round on the 8
answers, and +25.1 to +25.8% real speed, at 4.5 ms per round for its tables; below the count table.

Before any new architecture, a diagnosis tool was written (`nexus/trace_trees.py`) that builds
every tree on the same rounds and records, for each round, why the answer's path leaves the tree:
the block's end, the right token not among the 32 candidates, ranked first but the node budget
went elsewhere, ranked 2nd to 4th, ranked 5th or lower. It also records how far each tree gets
against DDTree on the same rounds. For DDTree on Draft-OPD (6,993 rounds, 7 answers): block end
7%, not a candidate 1%, ranked first not kept 9%, ranked 2 to 4 43%, ranked 5 or lower 40%. The
pair scores without the sureness numbers got less far than DDTree in 10 to 13% of rounds: they
damaged the path of first choices.

## Phase 10, November 20 to 27: the published second passes made here, and the first clear win

D2SD and D-Loop have no released weights. Their recipes were written into `nexus/train_loop.py`
(D-Loop: the same drafter retrained with prefix and suffix views, β = 0.7, τ = γ = 7; D2SD: a
separate drafter with the same views). The first D2SD training died at its first checkpoint when
the home folder filled; both are queued again (`loop3-dloop`, `loop4-d2sd`).

The second drafters were deepened (two and four of the drafter's layers, `opd-e2`, `opd-e4`) and
measured on the same rounds. The pair scores were given three numbers about the token before the
place: the first run's log-probability of it, whether it was the first run's first choice, and the
first choice's log-probability (`opd-p2s`, against `opd-p2` trained the same way without them).

The results, Draft-OPD, 63-node trees. Right after a fork, 12 validation answers: first run
20.7%, count table 39.5%, pair scores 42.4%, pair scores with the numbers 47.5%, one-layer second
drafter 49.1%, two layers 55.6%, four layers 57.3%. Tokens per round in replay on 7 to 8
validation answers: plain 5.72, DDTree 7.56, count table 8.31, pair scores 8.19, pair scores with
the numbers 8.67, one layer 8.21, two layers 8.35 (8.71 correcting every node), four layers 8.48
(8.94 every node). Real speed over plain on 3 long answers, the same text in every row: plain 5.80
tokens per round at 64.0 ms; DDTree 7.56 at 68.7 ms, +21.2%; pair scores (`opd-p1`) 8.26 at 72.5
ms, +25.6%; pair scores with the numbers 8.75 at 73.2 ms, +31.8%; one layer 8.29 at 75.3 ms,
+21.3%; two layers 8.51 at 84.7 ms, +10.7%; four layers 8.63 at 95.7 ms, −0.6%. The count table on
earlier runs: +28 to +30%.

What the trace says the three numbers do (6,993 rounds of DDTree, 7 answers). Stops with the right
token ranked 5th or lower fell from 28.3% to 24.9%; ranked 2nd to 4th stayed at 38.5%; ranked first
but the budget went elsewhere fell from 13.5% to 12.5%; block end rose from 9.9% to 11.9%; not a
candidate rose from 9.9% to 12.1% (the path goes further, so it meets more). Against DDTree, the
scores without the numbers got further in 41.9% of rounds and less far in 10.3% (gaining 0.87
tokens a round, losing 0.18); with the numbers, further in 49.4% and less far in 4.5% (gaining
1.16, losing 0.06). Keeping only the corrector's first choice at every node: DDTree 6.00, without
the numbers 6.03, with them 6.39. So about 70% of the gain is better ranking and 30% is less damage
to DDTree's good rounds, and both fit one mechanism: the numbers set the strength of the
correction, large after an unexpected token and small after an expected one.

## The current table (Draft-OPD, 63-node trees)

| Method | After a fork (12 ans.) | Tokens a round, replay (7–8 ans.) | Real speed over plain (3 ans.) | Extra cost a round |
| --- | --- | --- | --- | --- |
| Plain drafter | 20.7% | 5.72 | 0 | 0 |
| DDTree | 20.7% | 7.56 | +21.2% | 1.6 ms |
| Count table (DForkTree) | 39.5% | 8.31 | +28 to +30% (earlier runs) | 2.3 to 3.1 ms |
| Pair scores (`opd-p2`) | 42.4% | 8.19 | +25.6% (`opd-p1`) | 4.5 ms |
| Pair scores told how sure (`opd-p2s`) | 47.5% | 8.67 | +31.8% | 4.5 ms |
| Second drafter, one layer | 49.1% | 8.21 | +21.3% | 6.9 ms |
| Second drafter, two layers | 55.6% | 8.35 (8.71) | +10.7% | 16 ms |
| Second drafter, four layers | 57.3% | 8.48 (8.94) | −0.6% | 27 ms |

Caveats: 3 answers for speed, 7 to 8 for replay, eager timing. The confirmation on 30 answers and
on 40 standard-benchmark prompts is queued.

## What is new and what is known

Known before us, and confirmed by us: the hidden-state cause of the fork problem (DForkTree); a
second pass through the drafter not paying for itself (DForkTree's LoRA pass, our experiments 3
and 7); one or two tokens before a place carrying most of the information (Beyond Parallel
Blindness, DForkTree's counts); distilling a re-anchored drafter into a small head (DForkTree's
appendix); residual correction of the first run's scores and over-strong residuals hurting
(GRAFT); correctors reaching +9% real speed over DDTree at 64 nodes on Qwen3-4B with a one-token
head (TreeFlash, 2606.03819, which is the published comparator for what we have).

Ours, as far as three literature searches found: splitting a second draft's gain into the token,
the second run, and the states, on the same rounds; the per-cause accounting of why a tree's path
stops and the attribution of each method's gain to causes; and feeding the first run's confidence
about the token before a place, as continuous numbers, into a learned per-node scorer, with the
measured +0.45 tokens a round and the trace reading of where it comes from. Whether the last of
these is distinct from DForkTree's on-path flag (two fixed strengths) is the test running now.

One idea proposed on November 27 and withdrawn the same day: drafting the tree in the drafter's
first pass by giving extra positions the alternative token at a fork. The fork's alternative token
is an output of that pass, so it cannot be an input to it; every repair becomes a second pass,
which is what we measured. MFD (2611.13017) shows the positions would be nearly free (3.70 ms for
16, 4.24 ms for 64) but conditions on learned slots, not tokens, and gains 5 to 8%.

## Where the ceiling is

The published margins over DDTree with the same drafter: TreeFlash +9.1% speed at 64 nodes,
DForkTree +15.1% throughput, TreeSpark +9 to +16% tokens, Weaver +6 to +15% speed, UBTree +25.3%
with a retrained drafter. Add-on correctors top out around +9 to +15%; the larger gains come from
retraining the drafter with the tree in mind. Our pair scores with the numbers are +8.7% real speed
over DDTree on 3 answers (+14% tokens on 7), at parity with TreeFlash, with 4.5 ms of cost that the
published correctors do at under 1 ms.

The addressable headroom from the trace, for the pair scores with the numbers: ranked 2nd to 4th
38.6% of rounds, worth 0.80 tokens a round if fixed; ranked 5th or lower 24.9%, worth 0.43; ranked
first but the budget went elsewhere 12.5%, worth 0.28; not among the 32, 12.1%, worth 0.12; block
end 11.9%, not fixable with this block. The four-layer second drafter proves most of the ranking
part is reachable (it cuts the two ranking causes to 10.8% and 10.2%); it only loses on time.

## The plan, with its decision rules

Queued and running as of November 27, 04:19 UTC, in this order: the four ablations of the three
numbers (`abl1-top1`: only "first choice or not", which is DForkTree's flag; `abl2-logp`: that and
the log-probability; `abl3-scalar`: all three but one scale per token, a gate rather than a shape;
`abl4-seed`: all three again with another seed, to measure noise); the confirmation of real speed
on 30 long answers (`conf1-speed`: DDTree, count table, `opd-p2`, `opd-p2s`) and on 40 prompts of
GSM8K, MATH-500, HumanEval and MT-Bench with thinking off (`conf2-bench`); the cost of the search
done on the GPU in tensors of fixed shape and as a CUDA graph (`cost1-search`); the D-Loop and D2SD
trainings (`loop3-dloop`, `loop4-d2sd`).

Decision rules fixed before the results. If the flag alone (`abl1-top1`) reaches 8.5 or more of the
8.63 tokens a round, the contribution is DForkTree's two strengths learned, a small claim; if 8.4
or less, the graded number does the work. If the confirmation drops the margin over DDTree below
+5%, the speed claim is gone and the write-up says so. When the retrained drafters finish, a
retrained drafter with plain DDTree is compared with the corrector on the same answers; if it is
close to or above the corrector, the main line after the class moves to the drafter, because that
is where the published +25% comes from.

Next after these, each one job: learned temperatures over depth for the budget stops (0.28 tokens a
round at stake); 4 to 8 trigram candidates at nodes after a fork for the coverage stops (0.12); the
four-layer drafter as a teacher for the pair scores at four mixing weights against target labels
alone (the ranking headroom; nobody has published this comparison); the pair scores told the whole
branch's first-run probability rather than two tokens; a 127-node tree, since the check is flat up
to the tile. If the combination is at or above +12 to +15% over DDTree with the corrector under 1
ms, the result is at the workshop bar; otherwise it is a class report of a measured chain.

## Dates and constraints

The class report is due December 3. Everything must be copied off the cluster by December 28; the
account is deleted January 1, 2027. The home folder has a 32 GB quota (an old 6.6 GB environment
was removed on November 27 after it filled the folder and stopped the runner). The runner is
restarted with one line on nexusclass01; stale git locks are moved aside, nothing is deleted by
jobs. Job code copies live in `/tmp/specdec-code-$USER/`; large downloads go to scratch.

## Where the records are

- `results/SUMMARY.md`: September and October, one page (written November 3).
- `docs/WHY_A_SECOND_DRAFT_HELPS.md`: the swap, the second run, and the states, on the same rounds.
- `docs/SECOND_RUN_RESTART.md`: the restart at the break, the limit with real states, the stand-ins.
- `docs/LEARNING_DURING_THE_ANSWER.md`: long answers, the window, learning during the answer.
- `docs/DIRECTION.md`: the November plan and where it stood day by day, through the one-layer
  second drafter and the count table.
- `docs/SECOND_DRAFTER_RESEARCH.md`: the published second drafters, ours, the pair scores, the
  diagnosis, the published second passes made here, the win, and the three literature readings.
- `nexus/results/<job>/`: every job's status, answers and traces; `nexus/queue/<job>.sbatch`: what
  each job ran, with a comment saying why.
- Checkpoints in `~/specdec-ckpt` on the cluster: `warm-opd-e1/e2/e4/e1m1`, `pairs-opd-p1/p2/p2s`,
  `restart-opd-m0.pt`, and the drafters being trained.
