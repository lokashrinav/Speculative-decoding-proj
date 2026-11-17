# Can a drafter learn from its own draft?

Shrinav Loka and Frank. Speculative decoding on Qwen3 with the DFlash block drafter.

Speculative decoding lets a small drafter guess a block of tokens that the large target model verifies in one pass. A second drafting pass in the same round adds accepted tokens, but costs almost a full extra pass. This project asks what a second pass really adds and how to make one that pays.

## Where the project stands (November 17, 2026)

The current direction, the plan and an index of every experiment are in [`docs/DIRECTION.md`](docs/DIRECTION.md). In short:

- A second draft helps because it gives the target one more candidate token at the place where the first draft breaks. The second pass itself adds almost nothing on top of that token ([`docs/WHY_A_SECOND_DRAFT_HELPS.md`](docs/WHY_A_SECOND_DRAFT_HELPS.md)).
- What is missing after the break is the target's hidden states. With the right token at the break, the first draft's old tokens give 0.55 more accepted tokens, and the drafter with the target's hidden states gives 2.03.
- So we are building a second pass that restarts the drafter after the break, and studying what to give it in place of the hidden states it does not have yet. The research plan is in [`docs/SECOND_RUN_RESTART.md`](docs/SECOND_RUN_RESTART.md).

Real tokens per second on long thinking answers, Qwen3-4B on an RTX A5000, every setting writing the same text (16 answers per drafter, complete):

| Setting | Public drafter | Draft-OPD |
|---|---|---|
| Plain DFlash | 44.2 | 68.8 |
| The swap: the drafter's second-choice token at its four likely breaks, checked in the same pass, no second drafter pass | 50.2 (+14%) | 76.0 (+10%) |
| The drafter sees only the last 3,072 tokens (4,096 for Draft-OPD) | 54.8 (+24%) | 73.8 (+7%) |
| That window and the swap | 62.0 (+40%) | 81.5 (+18%) |
| Small add-on weights taught ahead of time | 59.6 (+35%) | 73.6 (+7%) |
| Those weights and the swap | 67.2 (+52%) | 81.6 (+19%) |

The window and trees of drafts are published ideas; see the direction document for what is and is not ours.

## What's here

| Folder | Contents |
|---|---|
| [`docs/DIRECTION.md`](docs/DIRECTION.md) | Where the project stands, what we are building, the plan, and where each experiment's code and results are. |
| [`docs/WHY_A_SECOND_DRAFT_HELPS.md`](docs/WHY_A_SECOND_DRAFT_HELPS.md) | The controlled study of why a second draft helps. |
| [`docs/LEARNING_DURING_THE_ANSWER.md`](docs/LEARNING_DURING_THE_ANSWER.md) | Long answers: learning during the answer, weights taught ahead of time, sharper attention, the window. |
| `nexus/` | The experiments that run on the class GPU cluster, their job scripts and their results (`nexus/results/`). |
| [`dflash/`](dflash/README.md) | The DFlash drafter we build on, real decoding with alternative drafts (`redraft.py`), the replay (`replay.py`), plus Frank's token-only secondary drafter. Install and usage notes there. |
| [`editor-experiment/`](editor-experiment/README.md) | The earlier experiment: a 3M-parameter editor trained to imitate a full second drafting pass. |
| [`results/`](results/SUMMARY.md) | One-page summary of everything tried up to early November, the D2SD paper note, and the run write-ups behind the numbers. |
| `docs/`, `.tours/` | Code tours for reading DFlash. |

## Earlier results (September and October, Qwen3-4B)

| | Extra accepted positions | Extra round cost |
|---|---|---|
| Full second drafting pass | +10.2% | about +30% |
| Best 3M editor | +3.7% | +5% to +6% |

The editor needed about +6% to break even and did not reach it. The second drafter repairs 79 and damages 33 blocks on its training prompts, but 35 and 41 on unseen prompts, so the editor is imitating prompt-specific repairs. A free 24-node tree from the drafter's own scores beats every editor.

## Setup

```
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[local]'
```

Checkpoints: `Qwen/Qwen3-4B` and `z-lab/Qwen3-4B-DFlash-b16`. The editor experiment needs a 24 GB GPU. Upstream DFlash is MIT licensed; see [`LICENSE`](LICENSE).
