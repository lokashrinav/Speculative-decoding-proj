"""Optional, untimed comparison with independent target-only generation."""

import os
import sys

import torch

# With DFLASH_MISMATCH=warn a mismatch is recorded and reported, not raised, so a
# benchmark can finish and report how often speculative output differs.
MISMATCHES: list[str] = []


def assert_same_tokens(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if torch.equal(actual, expected):
        return
    common = min(actual.shape[1], expected.shape[1])
    differences = (actual[:, :common] != expected[:, :common]).nonzero()
    position = int(differences[0, 1]) if differences.numel() else common
    message = (
        f"{label}: greedy token mismatch at position {position}; "
        f"sequence lengths {actual.shape[1]} vs {expected.shape[1]}"
    )
    if os.environ.get("DFLASH_MISMATCH") == "warn":
        MISMATCHES.append(message)
        print(f"WARNING: {message}", file=sys.stderr)
        return
    raise RuntimeError(message)


@torch.inference_mode()
def check_greedy_output(target, input_ids, output_ids, max_new_tokens, stop_token_ids):
    # Plain argmax decoding with its own cache and stopping logic. Using forward
    # directly avoids inheriting checkpoint generation penalties or EOS defaults.
    tokens = [input_ids]
    next_input = input_ids
    cache = None
    stops = torch.tensor(stop_token_ids or [], device=input_ids.device, dtype=torch.long)
    for _ in range(max_new_tokens):
        output = target(
            input_ids=next_input, past_key_values=cache, use_cache=True,
            logits_to_keep=1, output_hidden_states=False,
        )
        next_input = output.logits[:, -1:].argmax(dim=-1)
        tokens.append(next_input)
        cache = output.past_key_values
        if torch.isin(next_input, stops).any():
            break
    reference = torch.cat(tokens, dim=1)
    assert_same_tokens(output_ids, reference, "Independent target check")
