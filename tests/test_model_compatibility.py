"""Offline Qwen3 compatibility checks using a tiny, randomly initialized model."""

import json
import tomllib
from pathlib import Path

import pytest
import torch
from packaging.requirements import Requirement
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from kev.api import SystemOneRequest, to_answers, to_record
from kev.evaluate import load
from kev.model import SPECIAL, DecisionModel


def test_notebook_runtime_versions_satisfy_package_requirements():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    requirements = {
        requirement.name: requirement.specifier
        for requirement in map(Requirement, project["project"]["dependencies"])
    }
    for name, version in {"torch": "2.11.0", "numpy": "2.3.5", "peft": "0.19.1"}.items():
        assert version in requirements[name]


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(17)
    words = ["[UNK]", "[PAD]", *SPECIAL, "ticket", "refund", "login", "yes", "no"]
    vocabulary = dict(zip(words, range(len(words)), strict=True))
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=SPECIAL,
    )
    base = tmp_path / "qwen3"
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        attention_dropout=0.0,
    )
    Qwen3ForCausalLM(config).save_pretrained(base)
    tokenizer.save_pretrained(base)
    model = DecisionModel(str(base), tokenizer, "cpu", lora=2, head_dim=8).eval()
    # Exercise a nonzero adapter: a freshly initialized LoRA has zero B weights.
    with torch.no_grad():
        for name, parameter in model.lm.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    path = tmp_path / "checkpoint"
    model.lm.save_pretrained(path)
    torch.save(
        {
            "base": str(base),
            "base_revision": None,
            "lora": 2,
            "head_dim": 8,
            "head": model.head.state_dict(),
        },
        path / "head.pt",
    )
    return path


@pytest.fixture
def record():
    request = SystemOneRequest(
        state="ticket refund",
        questions={
            "assignment": {
                "type": "choice",
                "instructions": "ticket",
                "criteria": {"refund": "refund", "login": "login", "none": "no"},
            },
            "usable": {"type": "noul", "instructions": "ticket"},
        },
    )
    return to_record(request)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_qwen3_checkpoint_merge_and_api_answers(checkpoint, record, attention):
    rec, metadata = record
    tokenizer, unmerged = load(str(checkpoint), "cpu", merge=False, attn=attention)
    _, merged = load(str(checkpoint), "cpu", merge=True, attn=attention)
    encoding = merged.encode(tokenizer, rec, strict=True)
    expected, actual = unmerged.probs(encoding), merged.probs(encoding)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
        assert torch.isfinite(right).all()
        torch.testing.assert_close(right.sum(), torch.tensor(1.0))
    answers = to_answers([p.tolist() for p in actual], metadata)
    assert answers["assignment"]["choice"] in {"refund", "login", "none"}
    assert 0 <= answers["usable"]["noul"] <= 1
    json.dumps(answers, allow_nan=False)


def test_newer_adapter_default_metadata_loads(checkpoint, record):
    rec, _ = record
    tokenizer, original = load(str(checkpoint), "cpu")
    expected = original.probs(original.encode(tokenizer, rec, strict=True))
    config_path = checkpoint / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(peft_version="0.21.0", lora_ga_config=None, use_bdlora=False)
    config_path.write_text(json.dumps(config))
    tokenizer, model = load(str(checkpoint), "cpu")
    probabilities = model.probs(model.encode(tokenizer, rec, strict=True))
    assert [len(p) for p in probabilities] == [3, 2]
    for probability, reference in zip(probabilities, expected, strict=True):
        assert torch.isfinite(probability).all()
        torch.testing.assert_close(probability.sum(), torch.tensor(1.0))
        torch.testing.assert_close(probability, reference, atol=1e-5, rtol=1e-5)
    assert json.loads(config_path.read_text())["use_bdlora"] is False


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_qwen3_question_isolation_and_prefix_reuse(checkpoint, record, attention):
    rec, _ = record
    tokenizer, model = load(str(checkpoint), "cpu", attn=attention)
    encoding = model.encode(tokenizer, rec, strict=True)
    full = model.probs(encoding)
    isolated = [
        model.probs(model.encode(tokenizer, {**rec, "questions": [question]}))[0]
        for question in rec["questions"]
    ]
    cached, prefix = model.probs_and_prefix(encoding)
    reused = model.probs_with_prefix(encoding, prefix)
    reused_again = model.probs_with_prefix(encoding, prefix)
    for alternatives in zip(full, isolated, cached, reused, reused_again, strict=True):
        for value in alternatives[1:]:
            torch.testing.assert_close(alternatives[0], value, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        rows = [torch.softmax(logits, -1) for logits in model.forward_rows_batch([encoding])[0]]
    for expected, actual in zip(full, rows, strict=True):
        torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
