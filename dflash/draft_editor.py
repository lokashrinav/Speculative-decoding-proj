"""A small editor that improves a draft without running the drafter again.

This is the editor from editor-experiment/compact_second_pass_prototype.py
(Shrinav Loka and Frank, October 2026), packaged so it can be trained on real
data and used in decoding:

    first drafter pass  ->  saved states + draft tokens  ->  editor  ->  edited draft

The edited draft replaces the first draft: the target checks only the edited
one. The editor reads what the first pass already computed (its final states
for the 16 block positions) and the draft's own tokens. Two narrow attention
layers produce a low-rank correction to those states, and the drafter's frozen
output head turns the corrected states into tokens. The correction starts at
zero, so an untrained editor leaves the draft exactly as it was. About 3 million
parameters, against about 537 million for a second drafter pass.
"""

from types import SimpleNamespace

import torch
from torch import nn

from .model import _draft_value, _output_head, _raw_input_embeddings


class DraftEditor(nn.Module):
    """read_tokens: which of the draft's tokens the editor is shown.
        "all"    every draft token (it edits a complete draft)
        "first"  only the first draft token (the October design)
        "none"   no tokens
    read_states: whether it also reads the first pass's internal states.
    first_slot: the first draft position it may change. 1 means it may change
        every draft token; 2 keeps the first draft token as it is.
    """

    def __init__(self, hidden_size, rank=256, block_size=16, heads=4, layers=2,
                 read_tokens="all", read_states=True, first_slot=1):
        super().__init__()
        if read_tokens not in ("all", "first", "none"):
            raise ValueError("read_tokens must be 'all', 'first' or 'none'")
        if read_tokens == "none" and not read_states:
            raise ValueError("the editor has to read the draft's tokens, the states, or both")
        if first_slot not in (1, 2):
            raise ValueError("first_slot must be 1 or 2")
        self.config = dict(hidden_size=hidden_size, rank=rank, block_size=block_size, heads=heads,
                           layers=layers, read_tokens=read_tokens, read_states=read_states, first_slot=first_slot)
        self.read_tokens, self.read_states, self.first_slot = read_tokens, read_states, first_slot
        self.state_in = nn.Linear(hidden_size, rank, bias=False)
        self.token_in = nn.Linear(hidden_size, rank, bias=False)
        self.position = nn.Parameter(torch.zeros(block_size, rank))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(rank, heads, 2 * rank, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.up = nn.Parameter(torch.zeros(hidden_size, rank))

    def coefficients(self, states, token_embeddings):
        """The correction in the editor's own small space, for the positions it may change.
        states: the first pass's final states, (batch, block, hidden), anchor first.
        token_embeddings: embeddings of [anchor, draft tokens], same shape."""
        length = states.shape[1]
        x = self.position[None, :length].expand(states.shape[0], -1, -1)
        if self.read_states:
            x = x + self.state_in(states)
        if self.read_tokens == "all":
            x = x + self.token_in(token_embeddings)
        elif self.read_tokens == "first":
            x = x + self.token_in(token_embeddings[:, 1])[:, None]
        for layer in self.layers:
            x = layer(x)
        return x[:, self.first_slot:]

    def forward(self, states, token_embeddings):
        """Corrected states for the positions the editor may change."""
        return states[:, self.first_slot:] + self.coefficients(states, token_embeddings) @ self.up.t()


def save_editor(editor, path):
    torch.save({"config": editor.config, "state": {k: v.detach().cpu() for k, v in editor.state_dict().items()}}, path)


def load_editor(path, device, dtype):
    saved = torch.load(path, map_location="cpu")
    editor = DraftEditor(**saved["config"])
    editor.load_state_dict(saved["state"])
    return editor.to(device=device, dtype=dtype).eval()


class EditorImprover:
    """Plug-in for dflash_generate_redraft: the editor turns the first draft into the edited draft.

    replace=True (the intended use): the edited draft replaces the first draft.
    replace=False: the first draft is kept as well and the target checks both;
    only used to measure what a backup would be worth.
    """

    def __init__(self, editor, replace=True):
        self.editor = editor
        self.replace = replace
        self.folded = None  # output head folded into the editor: (vocabulary, rank)

    def start(self, first_drafter):
        pass

    def _logits(self, s, coefficients):
        """Edited logits. The editor only adds a low-rank correction, so the first
        pass's logits are reused and the large output head is not read a second time."""
        first = self.editor.first_slot
        head = _output_head(s.target)
        softcap = _draft_value(s.first_drafter.config, "final_logit_softcapping")
        plain_head = getattr(head, "bias", None) is None and not (softcap is not None and float(softcap) > 0)
        if not plain_head:
            corrected = s.draft_states[:, first:] + coefficients @ self.editor.up.t()
            return s.first_drafter.compute_logits(corrected, head)
        if self.folded is None:
            self.folded = (head.weight.float() @ self.editor.up.float()).to(coefficients.dtype)
        multiplier = float(_draft_value(s.first_drafter.config, "output_multiplier", 1.0))
        return s.draft_logits[:, first - 1 :] + (coefficients @ self.folded.t()) * multiplier

    def __call__(self, round_state: SimpleNamespace):
        s = round_state
        first = self.editor.first_slot
        if s.block_ids.shape[1] <= first:
            return None
        embeddings = _raw_input_embeddings(s.target, s.block_ids, 1.0).to(s.draft_states.dtype)
        coefficients = self.editor.coefficients(s.draft_states, embeddings)
        edited = s.block_ids.clone()
        edited[:, first:] = self._logits(s, coefficients).argmax(dim=-1)
        return edited
