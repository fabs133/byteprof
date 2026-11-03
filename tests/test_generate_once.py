"""generate_once against fake models that mimic what was observed on the real ones.

FakeBolmo reproduces Bolmo's remote generate(): the length criterion is applied to
a list of generated atoms that starts with the second-to-last prompt byte and whose
second entry is the re-decoded last prompt byte, so the stop point is
(input_len - 1) + max_new_tokens atoms and the real cap on new bytes is
input_len + max_new_tokens - 3, not max_new_tokens. The output is
input_ids[:, :-2] followed by that list.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
nn = torch.nn

from byteprof.config import ProfileConfig  # noqa: E402
from byteprof.counting import AlignmentError, cap_for  # noqa: E402
from byteprof.runner import generate_once  # noqa: E402

BOS = 1


class FakeTok:
    """Byte tokenizer: id = byte + 2, BOS = 1 prepended."""

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[BOS] + [b + 2 for b in text.encode()]])}

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return bytes(i - 2 for i in ids if i >= 2).decode("utf-8", errors="replace")


class FakeBolmo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.local_encoder = nn.Linear(1, 1)  # marker used for model-kind detection

    def generate(self, inputs, max_new_tokens=None, stopping_criteria=None, do_sample=False, **kw):
        max_len = (inputs.shape[1] - 1) + max_new_tokens
        atoms = [inputs[0, -2].item()]
        while True:
            # first step re-decodes the last prompt byte (forced decoding), then new bytes
            atoms.append(inputs[0, -1].item() if len(atoms) == 1 else ord("a") + len(atoms) % 26 + 2)
            t = torch.tensor(atoms).unsqueeze(0)
            if len(atoms) >= max_len:
                break
            if stopping_criteria is not None and bool(stopping_criteria(t, None).all()):
                break
        return torch.cat([inputs[:, :-2], torch.tensor([atoms])], dim=1)


class FakeHF(nn.Module):
    """Standard token model: one token = 3 bytes ('abc'), output = prompt + new tokens."""

    def generate(self, inputs, max_new_tokens=None, do_sample=False, **kw):
        new = torch.tensor([[b + 2 for b in b"abc"] * max_new_tokens])
        return torch.cat([inputs, new], dim=1)


class MisalignedModel(nn.Module):
    def generate(self, inputs, max_new_tokens=None, do_sample=False, **kw):
        return torch.cat([inputs.flip(1), inputs], dim=1)


def _cfg(n):
    return ProfileConfig(max_new_bytes=n, device="cpu")


PROMPT = "The history of the printing press shows how a single change can reshape society."


def test_bolmo_like_count_matches_cap_not_prompt_length():
    n, ms = generate_once(FakeBolmo(), FakeTok(), PROMPT, _cfg(32), lambda: None)
    assert n == 32  # buggy generate() alone would give len(PROMPT) + 29


def test_hf_like_counts_new_tokens_in_bytes():
    n, _ = generate_once(FakeHF(), FakeTok(), PROMPT, _cfg(5), lambda: None)
    assert n == 15


def test_misaligned_output_raises():
    with pytest.raises(AlignmentError):
        generate_once(MisalignedModel(), FakeTok(), PROMPT, _cfg(5), lambda: None)


def test_cap_for_units():
    assert cap_for("bytes", 32, max_token_bytes=1) == 32 + 3  # tolerate a cut multi-byte char
    assert cap_for("tokens", 10, max_token_bytes=7) == 70
