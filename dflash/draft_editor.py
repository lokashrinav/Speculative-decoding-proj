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

Two additions keep it from breaking tokens that were already right:
  confidence  the editor is told how sure the first pass was about each token
  margin      a token is changed only when the editor prefers its new token
              over the draft's token by at least this much (in logits)
"""

from types import SimpleNamespace

import torch
from torch import nn

from .model import _draft_value, _output_head, _raw_input_embeddings

CONFIDENCE_SIZE = 4


def draft_confidence(draft_logits):
    """How sure the first pass was about each draft token.
    draft_logits: (batch, block - 1, vocabulary), the first pass's logits for the draft tokens.
    Returns (batch, block, 4): top probability, runner-up probability, their gap and
    the entropy. The anchor (position 0) is a known token, so it counts as certain."""
    probabilities = draft_logits.float().softmax(dim=-1)
    top = probabilities.topk(2, dim=-1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
    features = torch.stack((top[..., 0], top[..., 1], top[..., 0] - top[..., 1], entropy / 5.0), dim=-1)
    anchor = features.new_tensor([1.0, 0.0, 1.0, 0.0]).expand(features.shape[0], 1, CONFIDENCE_SIZE)
    return torch.cat((anchor, features), dim=1)


class DraftEditor(nn.Module):
    """read_tokens: which of the draft's tokens the editor is shown.
        "all"    every draft token (it edits a complete draft)
        "first"  only the first draft token (the October design)
        "none"   no tokens
    read_states: whether it also reads the first pass's internal states.
    first_slot: the first draft position it may change. 1 means it may change
        every draft token; 2 keeps the first draft token as it is.
    normalize: rescale each input and the output so no single input drowns the others.
    confidence: also read draft_confidence(), the first pass's sureness per token.
    """

    def __init__(self, hidden_size, rank=256, block_size=16, heads=4, layers=2,
                 read_tokens="all", read_states=True, first_slot=1, normalize=False, confidence=False):
        super().__init__()
        if read_tokens not in ("all", "first", "none"):
            raise ValueError("read_tokens must be 'all', 'first' or 'none'")
        if read_tokens == "none" and not read_states:
            raise ValueError("the editor has to read the draft's tokens, the states, or both")
        if first_slot not in (1, 2):
            raise ValueError("first_slot must be 1 or 2")
        self.config = dict(hidden_size=hidden_size, rank=rank, block_size=block_size, heads=heads,
                           layers=layers, read_tokens=read_tokens, read_states=read_states, first_slot=first_slot,
                           normalize=normalize, confidence=confidence)
        self.read_tokens, self.read_states, self.first_slot = read_tokens, read_states, first_slot
        self.normalize, self.confidence = normalize, confidence
        self.margin = 0.0
        self.state_in = nn.Linear(hidden_size, rank, bias=False)
        self.token_in = nn.Linear(hidden_size, rank, bias=False)
        self.position = nn.Parameter(torch.zeros(block_size, rank))
        if normalize:
            self.state_norm = nn.LayerNorm(hidden_size)
            self.token_norm = nn.LayerNorm(hidden_size)
            self.in_norm = nn.LayerNorm(rank)
            self.out_norm = nn.LayerNorm(rank)
        if confidence:
            self.confidence_in = nn.Linear(CONFIDENCE_SIZE, rank)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(rank, heads, 2 * rank, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.up = nn.Parameter(torch.zeros(hidden_size, rank))

    def coefficients(self, states, token_embeddings, confidence=None):
        """The correction in the editor's own small space, for the positions it may change.
        states: the first pass's final states, (batch, block, hidden), anchor first.
        token_embeddings: embeddings of [anchor, draft tokens], same shape.
        confidence: draft_confidence() of the first pass, needed when the editor reads it."""
        length = states.shape[1]
        x = self.position[None, :length].expand(states.shape[0], -1, -1)
        if self.read_states:
            x = x + self.state_in(self.state_norm(states) if self.normalize else states)
        if self.read_tokens == "all":
            x = x + self.token_in(self.token_norm(token_embeddings) if self.normalize else token_embeddings)
        elif self.read_tokens == "first":
            first = token_embeddings[:, 1]
            x = x + self.token_in(self.token_norm(first) if self.normalize else first)[:, None]
        if self.confidence:
            if confidence is None:
                raise ValueError("this editor reads the first pass's confidence")
            x = x + self.confidence_in(confidence.to(x.dtype))
        if self.normalize:
            x = self.in_norm(x)
        for layer in self.layers:
            x = layer(x)
        if self.normalize:
            x = self.out_norm(x)
        return x[:, self.first_slot:]

    def forward(self, states, token_embeddings, confidence=None):
        """Corrected states for the positions the editor may change."""
        return states[:, self.first_slot:] + self.coefficients(states, token_embeddings, confidence) @ self.up.t()


def keep_unless_sure(logits, draft_tokens, margin):
    """The edited tokens. A draft token is changed only when the editor's logit for
    its new token beats its logit for the draft's token by more than the margin."""
    best = logits.argmax(dim=-1)
    if margin <= 0:
        return best
    gap = logits.gather(-1, best[..., None]) - logits.gather(-1, draft_tokens[..., None])
    return torch.where(gap[..., 0].float() > margin, best, draft_tokens)


def save_editor(editor, path, margin=0.0):
    torch.save({
        "config": editor.config, "margin": float(margin),
        "state": {k: v.detach().cpu() for k, v in editor.state_dict().items()},
    }, path)


def load_editor(path, device, dtype):
    saved = torch.load(path, map_location="cpu")
    editor = DraftEditor(**saved["config"])
    editor.load_state_dict(saved["state"])
    editor.margin = float(saved.get("margin", 0.0))
    return editor.to(device=device, dtype=dtype).eval()


class EditorImprover:
    """Plug-in for dflash_generate_redraft: the editor turns the first draft into the edited draft.

    replace=True (the intended use): the edited draft replaces the first draft.
    replace=False: the first draft is kept as well and the target checks both;
    only used to measure what a backup would be worth.
    margin: see keep_unless_sure(); by default the value stored with the editor.
    """

    def __init__(self, editor, replace=True, margin=None):
        self.editor = editor
        self.replace = replace
        self.margin = float(getattr(editor, "margin", 0.0)) if margin is None else float(margin)
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
        confidence = None
        if self.editor.confidence:
            confidence = draft_confidence(s.draft_logits).to(s.draft_states.dtype)
        coefficients = self.editor.coefficients(s.draft_states, embeddings, confidence)
        edited = s.block_ids.clone()
        edited[:, first:] = keep_unless_sure(self._logits(s, coefficients), s.block_ids[:, first:], self.margin)
        return edited
