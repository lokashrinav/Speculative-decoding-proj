# Gate 1 on Modal: a predictor of the target's deep states for a restarted drafter

Instructions for whoever runs this on Modal (written November 27, 2026). Everything below is exact:
file paths, commands, the numbers to reproduce, and the rule for stopping.

## 1. What is being tested, in one paragraph

A DFlash drafter writes a block of 16 tokens reading the target's hidden states of the text so
far, taken from five target layers (1, 9, 17, 25, 33 of Qwen3-4B's 36), concatenated (5 × 2560
= 12,800 numbers per token) and projected by the drafter's `fc` layer into one 2,560-number vector.
When the drafter is restarted inside a block (a second run after a break), the new tokens have no
such states, because the target makes them only in its check. With the real states handed in, the
restart earns about four times the tokens it earns with none. The real states of the two deep
layers alone (25 and 33) give 94% of that; the three shallow layers alone are worse than nothing.
Gate 1 asks: can a small network predict the deep two layers' states well enough that the restart
earns most of the real-state gain? The current predictor (`opd-m0`) gets about a third of it.

## 2. What already exists and what was already tried (do not repeat)

- `dflash/stand_in.py`: `StatePredictor` (outputs ONE 2,560-number vector, the `fc`-projected
  state, not per-layer states), `predicted_states`, `project`, `as_context`.
- `nexus/train_restart.py`: its trainer. `--loss vectors` is a plain vector error against the real
  projected vector. `--loss tokens` (the default) is END-TO-END through the frozen drafter: the
  likelihood of the answer's real tokens after a restart made with the predicted vectors, plus
  `--vector-weight` times the vector error. `--layers N --memory M` adds N transformer layers that
  look at the last M real vectors.
- `nexus/restart_probe.py`: the measurement. `--more` adds the rows `deep1`, `deep2` (real states of
  the last one or two layers, the rest repeated from the last real vector), `shallow1..4`.
- Already measured on Draft-OPD, the same 12 validation answers (18,316 rounds), tokens accepted
  after the right token at the break: nothing 1.02; `opd-m0` (vector loss, 366 answers, 11.0M
  parameters) 2.20; `opd-c0` (end-to-end token loss, 659 answers, 12.9M) 1.90; `opd-s0`
  (end-to-end, 2 layers over the last 8 real vectors, 40.7M) 1.71; real states 4.09. So the
  end-to-end loss and the small memory transformer were WORSE than the vector loss here. The
  reference set for Gate 1 is the 30-answer run with `--more` (`probe-more-opd-0`): nothing 0.97,
  `opd-m0` 2.01, deep2 (real 25 and 33 only) 3.48, all five real 3.69.
- Not yet tried, and the point of Gate 1: (a) predict only the deep two layers' RAW states
  (2 × 2560), with the shallow three taken from the last real vector exactly as the probe's `deep2`
  row does, then `fc`; (b) train on all the data (`long1` train + `think1`, about 8,000 answers)
  instead of 366 to 659; (c) a bigger predictor, since the data is 20 times larger.

## 3. Files to copy from Nexus (the home folder is `/fs/classhomes/sloka`)

From your laptop (Duo prompt as usual), then `modal volume put` into a volume named, say, `specdec`:

```
ssh sloka@nexusclass01.umiacs.umd.edu "tar czf - specdec-data/long1 specdec-data/think1" > specdec-data.tgz      # about 180 MB
scp sloka@nexusclass01.umiacs.umd.edu:specdec-ckpt/restart-opd-m0.pt .                                          # 18 MB, for parity
```

The code: clone `github.com/lokashrinav/speculative-decoding-proj`, branch `nexus-jobs`, commit
`acb2b018` or later. The models come from Hugging Face on Modal: `Qwen/Qwen3-4B` (target) and
`bingyang-lei/Qwen3-4B-Thinking-Draft-OPD` (the drafter, "opd"). Set `HF_HOME` to a volume path so
they download once (about 9 GB).

Python environment: `pip install -e .` in the repository (the `pyproject.toml` lists the
dependencies), plus `safetensors`, `datasets`. Torch with CUDA. Nothing else is needed.

On the volume the layout the scripts expect:

```
/vol/specdec-data/long1/     (shards *.pt; the split is a hash of the prompt, built in)
/vol/specdec-data/think1/
/vol/specdec-ckpt/restart-opd-m0.pt
/vol/hf-cache/               (HF_HOME)
```

Pass `--data-dir /vol/specdec-data/long1 --more-data /vol/specdec-data/think1 --adapter-dir
/vol/specdec-ckpt --results-dir /vol/results/<name>` where the scripts take them (`train_restart.py`
has `--data-dir`, `--adapter-dir`; check whether it has `--more-data` and add it as in
`nexus/train_pairs.py` if it does not, which is a few lines: load both folders' training answers).

## 4. Step 0: parity (30 minutes on one A100-80GB)

Run the probe exactly as the November 18 reference, 30 validation answers:

```
python nexus/restart_probe.py --draft opd --data-dir /vol/specdec-data/long1 --results-dir /vol/results/parity \
  --part validation --max-answers 30 --max-hours 1.5 --predictor opd-m0 --more --adapter-dir /vol/specdec-ckpt
```

Read `status.json` → `summary` → `all`: `none` must be about 0.97, `predicted` about 2.01, `deep2`
about 3.48, `real` about 3.69 (a different GPU can move these by a few hundredths, not more). If
they do not match, stop and report; nothing else is meaningful until they do.

## 5. Gate 1: the deep-layer predictor (one A100-80GB, about 4 to 6 hours)

The code change (in `dflash/stand_in.py` and `nexus/train_restart.py`), kept minimal:

1. `StatePredictor(..., deep_only=True)`: `self.out = nn.Linear(hidden, 2 * layer_size)` where
   `layer_size = 2560` (one target layer's part of the raw 12,800-number state). Record
   `deep_only` in `sizes` so `load_predictor` rebuilds it.
2. A function `assemble(drafter, deep_raw, last_real_raw)`: a raw state [count, 12,800] whose last
   two layer parts (indices 3 × 2560 to 5 × 2560) are the prediction and whose first three parts
   are copied from the last real raw state before the round, exactly as `restart_probe.py` builds
   `deep2` (lines near "for count in (1, 2)"). Then `project(drafter, assembled)`.
3. `predicted_states`: with `deep_only`, return `assemble(...)` projected, instead of
   `as_context(drafter, predictor(...))`. The predictor's INPUTS stay as they are (the drafter's
   final state at the token, at the token after, the token's embedding, the last real projected
   vector, the place, the coming token).
4. Training targets: with `--loss vectors`, the vector error is against the real raw deep parts
   (2 × 2560), not the projected vector. With `--loss tokens`, the token loss goes through
   `assemble` → `project` → the frozen drafter as before. Both need the raw real states of the
   tokens, which the training already has (they come from `target_features`; keep the raw
   features, not only the projected ones).

Train two predictors, in parallel on two GPUs if available, all data, one pass each:

```
python nexus/train_restart.py --draft opd --data-dir /vol/specdec-data/long1 --more-data /vol/specdec-data/think1 \
  --results-dir /vol/results/gate1-vectors --adapter-dir /vol/specdec-ckpt --save opd-d1v --deep-only \
  --loss vectors --hidden 2048 --inner 768 --rate 3e-4 --max-hours 4 --check-answers 30 --part train
python nexus/train_restart.py --draft opd --data-dir /vol/specdec-data/long1 --more-data /vol/specdec-data/think1 \
  --results-dir /vol/results/gate1-tokens --adapter-dir /vol/specdec-ckpt --save opd-d1t --deep-only \
  --loss tokens --vector-weight 1.0 --hidden 2048 --inner 768 --rate 3e-4 --max-hours 4 --check-answers 30 --part train
```

(`--deep-only` is the new flag from the change above; `--hidden 2048 --inner 768` is about 4× the
old predictor; both must fit easily on an 80 GB card with the target and the drafter loaded.)

Each run ends by measuring the restart on `--check-answers` validation answers and writes
`restart_after_the_break` into its `status.json`. Then run the 30-answer probe with each saved
predictor for the exact comparison with Step 0:

```
python nexus/restart_probe.py --draft opd --data-dir /vol/specdec-data/long1 --results-dir /vol/results/gate1-probe-v \
  --part validation --max-answers 30 --max-hours 1.5 --predictor opd-d1v --more --adapter-dir /vol/specdec-ckpt
```

(and the same with `opd-d1t`). Also time one predictor call for 4 and for 16 tokens with CUDA
events after warm-up, and report the milliseconds.

## 6. The rule

Report, for each predictor, `predicted` from the 30-answer probe next to 0.97 / 2.01 / 3.48 / 3.69,
and the milliseconds per call. Gate 1 clears only if a predictor reaches 2.70 or more (closing
about 70% of the way from nothing to the deep-two ceiling) at a few milliseconds per call. Between
2.2 and 2.7 is "more data and the deep-only target help, but not enough"; report and stop. Do not
build Arch 2 (the target's own layers with bridge adapters) unless Gate 1 clears; it is a day of
new code and the restart's speed ceiling with REAL states on our stack is already below the
pass-free corrector (a 13 ms restart pass at four breaks: +17% real speed over plain, against +26%
for DDTree and +33% for the corrector), so a predictor only pays if it is used as an input to the
corrector, which is Gate 2 and needs the Nexus code for the pair scores (`dflash/pair_scores.py`,
`nexus/train_pairs.py`).

## 7. Things not to do

- Do not use the 12-answer numbers (1.02 / 2.20 / 4.09) as the reference; they are a different
  answer set. The 30-answer set with `--more` is the reference.
- Do not train on validation answers; `--part train` and the hash split keep them apart.
- Do not change the probe; it is the measurement everything else is compared against.
- Report numbers with their answer count and the job that produced them.
