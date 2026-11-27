Overview

A drafter guesses a block of 16 tokens and the target checks them in one pass. We asked whether a second look at the draft could speed decoding up. We found that a second run of the drafter helps mainly by handing the target one more candidate at the place where the draft breaks, that the ending it writes after that token is bad because the target's hidden states for the new tokens do not exist yet, and that no second run, predictor of those states, or teacher of a small network earns back its cost in real time. What does pay is a small network that corrects the scores of a tree built from the first draft, told whether the token before each place was the drafter's own first choice: it is 6.3% faster than DDTree on 30 long answers in plain PyTorch, about 12% faster with its work recorded as a CUDA graph on 4 answers, and it keeps its gain on code and chat, where a published count table built on math text falls below DDTree.

Configurations for Experiments:
GPU: RTX A5000
Target: Qwen3-4B, thinking mode, greedy decoding
Drafter: the public DFlash checkpoint (z-lab/Qwen3-4B-DFlash-b16), 5 layers, 16-token block, not retrained.
Baseline/Plain Draft: 3.31 accepted tokens per round
Data: answers the target wrote in thinking mode to prompts from AIME (before 2024), MATH level 5, LongWriter and UltraChat, up to 16,384 tokens each (the long1 set: 366 training answers, 100 validation answers), plus about 8,000 shorter answers of the same kinds for training only (think1). Every answer is assigned by a hash of its prompt to a training, validation or test split (60/20/20); everything trained learned from training answers, every number below is on validation answers, and the test split has not been used.
Out-of-domain suite: 40 prompts of the standard benchmarks with thinking off, 10 each of GSM8K, MATH-500, HumanEval and MT-Bench, the target's own answers.
Rule: greedy and lossless, so every method writes exactly the target's text; no oracle input in any speed measurement. Plain-draft acceptance differs between subsets of answers (3.31 on 50 public-drafter answers, 4.82 on 16 long Draft-OPD answers, 5.22 on 30, 5.72 on the 8 replay answers), so every comparison below is between methods on the same answers and the same text.

What happens when you run a second drafter again?

To run a second draft, we take the spots in the first draft where it's mostly likely to be incorrect and rerun the drafter from those points (typically limited to 4 spots). In our first basic experiment, we used the same drafter model again and with four cut points, we obtained an 8.4% acceptance for a double cost. D2SD ran the same experiments and we obtained similar results.

By retraining the drafter, compared to our baseline, D-Loop obtains +56% accepted tokens where we obtained +8.4%, for a similar relative cost: both methods roughly double the drafter's time per round (theirs +124%, ours about +140%), while the target's check time stays about the same. For D-Loop, the extra tokens outweigh the extra time, giving a +32% speedup.

By training a separate second drafter, compared to our baseline, D2SD obtains +33% accepted tokens.

—

We found that the second run costs more time than its gain is worth. So our question became: can a smaller model give the same or comparable gain at a fraction of the time?

In this manner, we've delved into a few experiments:

Experiment 1. What is the second run's gain made of?

(The +8.4% above is the second run left free: shown its own first draft, it usually repeats its first choice at the break. The +20.2% below is the second run given the drafter's second choice at the break and rewriting the rest. Both are on the same 50 answers.)

The second run does two things at once: it puts a new token at the break, and it writes a new ending after that token. To measure each part on its own, we ran three settings on the same 50 answers (public drafter): the plain draft; the plain draft with the drafter's second choice swapped in at the four likely breaks, and nothing written after it; and the full second run, which swaps the token and rewrites the ending.

In our experiment, the plain draft accepted 3.31 tokens per round. With the drafter's second choice swapped in at the four likely breaks and nothing written after it, 3.93 tokens were accepted per round, +18.8%. With the full second run, which swaps the token and rewrites the ending, 3.98 tokens per round, +20.2%.

Most of the gain came from the swap rather than the second run.

The conclusion leaves one question: is the rewritten ending worth so little because a good ending cannot add much, or because this second run writes it badly? The two answers lead to different work. If endings cannot add much, the second run is a dead end and the swap is the method. If the second run writes them badly, the ending is worth fixing.

Experiment 2. Does the second run write the ending badly, or is the ending not worth much?

To settle it, we took answers the target had already written, ran the drafter over them, and collected every round where the draft broke, since there we knew the right token. In each of those rounds we restarted the drafter from the right token in two ways. First, as a second run in the same round, with the new tokens given as tokens only. Second, as the next round's draft, with the target's hidden states for the new tokens, which the target produces when it checks them. Then we counted the tokens the target accepted after the restart.

In the same round, the restart added 0.59 tokens on the public drafter and 0.93 on Draft-OPD. With the target's hidden states, it added 2.17 and 3.65. Same drafter, same starting point, nearly four times the result.

So the second run writes the ending badly, and the ending is worth fixing. The cause is a single missing input: the target's hidden states for the new tokens. They cannot exist in the same round, because the target makes them in the check, after the drafting.

The question for the rest of the project follows from this: how do we replace those hidden states within the same round, at a cost below what they earn?

Experiment 3. Can a small network predict the missing hidden states?

Experiment 2 showed the second run is bad because the new tokens have no hidden states yet. The direct fix is to make them. We trained a small network that takes what the drafter already has, the states for the text before the block and the token g itself, and tries to output the hidden state the target would have made for g.

Then we ran the same test as experiment 2. At every round where the draft broke, we did the second run from the right token three times, with three inputs: no hidden state for the new tokens, the predicted one from our network, and the real one taken from the target's later check, which gives the ceiling. We counted how many of the new tokens the target accepted.

On Draft-OPD, over 30 validation answers, the restart adds 0.97 accepted tokens after its first token with nothing, 2.01 with the predicted states, and 3.69 with the real ones. So the predictor closes about a third of the gap.

A predictor that closes a third of the gap is not enough, because the second run still has to be paid for. We measured that in real decoding on Draft-OPD, 16 long answers, every method writing the same text. The plain draft accepted 4.82 tokens per round. The second run at the four likely breaks, with the predicted states, accepted 6.24 tokens per round, 30% more, but it cost 14.7 ms on a round of about 70 ms, so the real speed was only 4% higher. DDTree, which builds a tree of 63 candidates from the first draft's own scores and costs 1.6 ms, accepted 6.59 tokens per round and was 26% faster. The second run on top of DDTree accepted 7.14 tokens per round and was 12% faster, which is slower than DDTree alone.

From here on, DDTree is the baseline. The drafter is Draft-OPD (bingyang-lei/Qwen3-4B-Thinking-Draft-OPD), which is trained for thinking-mode text, and the plain draft on it accepts 5.72 tokens per round on our validation answers. A method has to beat DDTree in real speed, not the plain draft, and the question is no longer how to make a second run good but how to get its gain without its pass.

Experiment 4. Can a second run that costs one layer do the job?

A whole second run costs about 1.7 ms per layer of the drafter and hardly anything per restart it holds, so the cheapest second run is one layer. We trained one layer of the drafter to restart at the break, reading the tokens of the branch, on 366 training answers, and measured it where the first draft is weakest: the place right after a fork, where the branch has just taken a token that was not the first draft's first choice. The first draft is right there 21% of the time. The one-layer second run is right 49% of the time. A layer started from new weights, not the drafter's, gets two thirds of that, so the training is what makes a few hundred answers enough.

Run as a second pass over a 63-node tree, the one layer accepts 8.21 tokens per round where DDTree accepts 7.56. But the pass costs 6.9 ms, and in real speed it is 21% faster than the plain draft, the same as DDTree. So one layer is as far as a second run can be cut, and it still does not pay.

Experiment 5. What does a free correction get?

DForkTree, a published method, corrects the tree's scores with no pass at all. It counts, in training text, which token the target writes after which two tokens, and lifts the score of a candidate that often follows the two tokens before it. We wrote it from the paper. Right after a fork it is right 39.5% of the time. On the tree it accepts 8.31 tokens per round and is 28 to 30% faster than the plain draft, against DDTree's 21%. Every learned correction we put on top of the counts gained tokens and lost speed, so the count table became the bar any learned method had to beat.

Experiment 6. Can a small network learn the correction instead of looking it up?

The count table needs stored text and fails where the text does not cover the task. We replaced it with a small network that scores a candidate from the two tokens before it, the drafter's state at the place, and the candidate's row of the output head, trained on the target's own answers. It is right after a fork 42.4% of the time, accepts 8.19 tokens per round, and gets less far than DDTree in 10% of rounds, because it corrects the first draft's good guesses as much as its bad ones.

Telling the network whether the token before the place was the first draft's own first choice fixed that. With that input it is right after a fork 47.5% of the time and accepts 8.67 tokens per round on the same 8 answers. In real decoding on 30 long answers it is 32.9% faster than the plain draft, against 25.0% for DDTree and 30.7% for the count table, at a cost of 4.4 ms per round for the scoring. Recording the whole scoring and tree build as one CUDA graph brings that cost to 1.7 ms with the same tokens; on 4 answers the graphed version is 34.7% faster than plain against 19.7% for DDTree, about 12% over DDTree, a number that is preliminary until the graphed version is timed on the full 30 answers and the benchmark prompts.

We also gave the network the first draft's probability of the token before and of its first choice, as numbers. They add nothing: trained four ways and with two seeds, the versions with and without the numbers are within 0.04 tokens per round of each other. The whole gain is the yes-or-no flag, which is the same distinction DForkTree makes with two fixed weights. So the on-path and off-path gating itself is not new; it is a learned form of DForkTree's two-weight split. What is new is replacing the stored count table with a small network over the drafter's own states and output embeddings, which needs no stored text and transfers out of domain.

Where the stored text matters is out of domain. On 40 prompts of the standard benchmarks with thinking off, the network is 23.8% faster than plain, the count table 20.6%, DDTree 15.1%. On HumanEval, which is code, the count table falls below DDTree (28.6% against 32.9%) while the network accepts 7.1% more tokens per round than DDTree and ties it in speed only because the 4.4 ms scoring was not graphed in that run.

Experiment 7. Why does the answer's path leave each tree?

To see what was left, we built every tree on the same rounds and recorded why the answer's path stopped in each: the block ended; the right token was not among the first draft's 32 candidates; it was ranked first but the tree's budget went elsewhere; it was ranked second to fourth; it was ranked fifth or lower. For the network with the flag, on 7,900 rounds: block end 12%, not a candidate 12%, ranked first but budget elsewhere 12.5%, ranked second to fourth 39%, ranked fifth or lower 25%.

The same accounting says what the flag did. Compared with the network without it, the stops with the right token ranked fifth or lower fell from 28% to 25%, and the rounds where the network got less far than DDTree fell from 10% to 4.5%. About 70% of its gain is better ranking and 30% is less damage to DDTree's good rounds.

It also says what a bigger second run knows. A four-layer second run correcting every node is right after a fork 57% of the time and accepts 8.98 tokens per round, 0.31 more than the network, but it costs 27 ms and is slower than the plain draft. On the same rounds it gets further than the network in 30% of rounds and less far in 17%; it makes large repairs and does real damage, mostly by spending the budget on the wrong branches.

Experiment 8. Can the network learn what the four-layer second run knows?

Two ways were tested. First, on the same rounds, we gave the network the four-layer run's ordering of the candidates but its own strengths, and the reverse. The ordering alone keeps almost none of the four-layer run's gain (8.69 against 8.67 tokens per round) and the strengths alone do harm (8.52); only the two together carry it. So a loss that teaches the order would not transfer the knowledge.

Second, we trained the network to match the four-layer run's probabilities of the candidates, mixed with the target's own labels at weights 0.25, 0.5 and 1, with two passes over the training text each. The results are 8.70, 8.67 and 8.51 tokens per round against 8.67 and 8.70 for the network trained on the target's labels alone with two seeds. The teacher adds nothing, and the teacher alone is worse. For a corrector this small, the target's own answers are as good a teacher as a second run that sees the branch.

Experiment 9. Can the drafter branch inside its own pass?

We also asked whether the drafter could write the branches after its likely breaks inside its first pass, reading its own alternatives from an early layer, to get a second run's continuations without a second pass. Three checks before building it: an untrained early readout is poor (at layer 3 of 5 the chain of first choices is right for 1.5 tokens against 5.3 at the end); the cost is affordable (about 3 ms); and an oracle that gives the four branches perfect three-token continuations raises DDTree from 7.57 to 8.27 tokens per round inside the same row budget, below the 8.67 the network already reaches. With perfect six-token continuations the oracle reaches 9.32, but a drafter's continuation is worth about a third of a perfect one (Experiment 2), which lands below the network again. We stopped it there, without training.

Experiment 10. Does a retrained drafter do better than a corrector?

D2SD and D-Loop retrain a drafter for the second pass and have no released weights. We trained D2SD's separate second drafter with its recipe on the cluster, on 7,690 answers (D-Loop did not fit in memory). Then we ran Experiment 2's test with it: at every round where the draft broke, restarted from the right token, we counted the tokens the target accepted after it. With tokens only, the trained second drafter adds 1.59 tokens where the original drafter added 0.83; with the target's real hidden states it adds 3.27 where the original added 3.65. So training the second drafter for the job closes about a third of the gap that the missing states leave, and the states still double what the trained pass gets.

In real decoding on 7 long answers, the trained second drafter at the four likely breaks accepts 7.19 tokens per round, 22% more than the plain draft's 5.88, but its pass costs 14 ms on a 65 ms round, and it is 0.7% faster than plain. DDTree on the same answers is 21.3% faster, the count table as a CUDA graph 32.3%, and the network with the flag as a CUDA graph 36.3% (8.89 tokens per round at 72.1 ms). D2SD's own paper reports a 20% speedup on a serving stack where its second pass costs about 4 ms; on our stack the same tokens cannot pay for a 14 ms pass. The retrained second drafter is below DDTree alone in both tokens and speed, so the pass-free corrector stays the method.

Main results

The methods side by side, all on Draft-OPD. Long thinking answers: tokens per round on the 8 replay answers, and real speed over the plain draft on the 30 long answers (plain PyTorch) with the graphed numbers on 4 answers in brackets. Benchmark prompts: real speed over plain on the 40 prompts with thinking off, plain PyTorch.

| Method | Tokens a round (8 answers) | Speed over plain, 30 long answers | Speed over plain, 40 benchmark prompts |
| --- | --- | --- | --- |
| Plain draft | 5.72 | 0 | 0 |
| DDTree (63 nodes) | 7.56 | +25.0% [+19.7%] | +15.1% |
| Count table (DForkTree) | 8.31 | +30.7% [+30.1% graphed] | +20.6% |
| One-layer second run | 8.21 | +21.3% (3 answers) | not run |
| Four-layer second run, every node | 8.98 | −0.6% (3 answers) | not run |
| Pair network without the flag | 8.19 | +28.1% | not run |
| Pair network with the flag | 8.67 | +32.9% [+34.7% graphed] | +23.8% |

What we know

A second draft helps because it hands the target one more candidate at the break; the second run itself adds about a tenth of that unless it has the target's hidden states for the new tokens, which cannot exist in the same round. A predictor of those states closes a third of the gap, and a second run at any depth loses in real speed to a free tree built from the first draft. The cheapest correction that beats that tree is a small network that scores a candidate from the branch's last two tokens and the drafter's state, told whether the token before was the drafter's own first choice: 6.3% faster than DDTree in eager timing on 30 long answers, about 12% with the scoring recorded as a CUDA graph on 4, and it transfers to code and chat where a count table built on math text does not. What it cannot learn is the knowledge of a second run that sees the branch, by ordering or by probabilities; that knowledge comes with the branch's tokens inside the pass, and it does not pay for its pass on this target.

What is still running

The staleness inputs (how far and how often the branch has left the first draft's path) as a learned strength of the correction [pending]; where the missing tokens sit in the first draft's vocabulary, which decides whether widening the candidates or a token-only proposer recovers the 12% of coverage stops [pending]; a 127-node tree where the target's check is flat [pending]; and the graphed network on the 40 benchmark prompts [pending].
