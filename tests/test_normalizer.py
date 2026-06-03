"""Tests du normalizer SFT et DPO."""
import json
from pathlib import Path

from src.data_pipeline.normalizer import (
    MAX_INSTRUCTION_WORDS,
    MIN_RESPONSE_WORDS,
    normalize_dpo_file,
    normalize_sft_file,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── SFT ───────────────────────────────────────────────────────────────────────

class TestNormalizeSft:

    def test_valid_record_is_kept(self, tmp_path):
        raw = [{"question": "What is hypertension?", "answer": "High blood pressure condition requiring treatment.", "language": "en"}]
        inp, out, audit = tmp_path / "medquad_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        assert normalize_sft_file(inp, out, audit) == 1
        records = _read_jsonl(out)
        assert records[0]["instruction"] == "What is hypertension?"
        assert records[0]["task_type"] == "medical_qa"

    def test_schema_fields_present(self, tmp_path):
        raw = [{"question": "Q?", "answer": "Answer with enough words here.", "language": "en"}]
        inp, out, audit = tmp_path / "chatdoctor_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        normalize_sft_file(inp, out, audit)
        r = _read_jsonl(out)[0]
        for field in ["id", "source", "language", "task_type", "instruction", "response", "metadata"]:
            assert field in r
        for meta in ["symptoms", "antecedents", "constantes", "transformation_ids", "split"]:
            assert meta in r["metadata"]

    def test_empty_instruction_filtered(self, tmp_path):
        raw = [{"question": "", "answer": "Valid long answer here.", "language": "en"}]
        inp, out, audit = tmp_path / "test_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        assert normalize_sft_file(inp, out, audit) == 0

    def test_response_too_short_filtered(self, tmp_path):
        raw = [{"question": "Q?", "answer": " ".join(["x"] * (MIN_RESPONSE_WORDS - 1)), "language": "en"}]
        inp, out, audit = tmp_path / "test_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        assert normalize_sft_file(inp, out, audit) == 0

    def test_instruction_too_long_filtered(self, tmp_path):
        raw = [{"question": " ".join(["w"] * (MAX_INSTRUCTION_WORDS + 1)), "answer": "Valid long answer here.", "language": "en"}]
        inp, out, audit = tmp_path / "test_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        assert normalize_sft_file(inp, out, audit) == 0

    def test_audit_log_written(self, tmp_path):
        raw = [{"question": "Q?", "answer": "Answer with enough words here.", "language": "en"}]
        inp, out, audit = tmp_path / "medquad_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        normalize_sft_file(inp, out, audit)
        entries = _read_jsonl(audit)
        assert len(entries) == 1
        assert entries[0]["operation"] == "normalization"

    def test_transformation_id_format(self, tmp_path):
        raw = [{"question": "Q?", "answer": "Valid long answer with words.", "language": "en"}]
        inp, out, audit = tmp_path / "chatdoctor_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        normalize_sft_file(inp, out, audit)
        r = _read_jsonl(out)[0]
        assert r["metadata"]["transformation_ids"][0].startswith("norm_chatdoctor_")

    def test_multiple_records(self, tmp_path):
        raw = [{"question": f"Q{i}?", "answer": "Valid answer with several words here.", "language": "en"} for i in range(5)]
        inp, out, audit = tmp_path / "medquad_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        assert normalize_sft_file(inp, out, audit) == 5

    def test_split_is_none(self, tmp_path):
        raw = [{"question": "Q?", "answer": "Long enough answer right here.", "language": "en"}]
        inp, out, audit = tmp_path / "medquad_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        normalize_sft_file(inp, out, audit)
        assert _read_jsonl(out)[0]["metadata"]["split"] is None


# ── DPO ───────────────────────────────────────────────────────────────────────

class TestNormalizeDpo:

    def test_valid_dpo_record(self, tmp_path):
        raw = [{"prompt": "Best treatment?", "chosen": "Use metformin.", "rejected": "Use nothing."}]
        inp, out, audit = tmp_path / "dpo_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        assert normalize_dpo_file(inp, out, audit) == 1
        r = _read_jsonl(out)[0]
        assert r["chosen"]["role"] == "assistant"
        assert r["chosen"]["content"] == "Use metformin."
        assert r["rejected"]["content"] == "Use nothing."

    def test_empty_prompt_filtered(self, tmp_path):
        raw = [{"prompt": "", "chosen": "Good.", "rejected": "Bad."}]
        inp, out, audit = tmp_path / "dpo_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        assert normalize_dpo_file(inp, out, audit) == 0

    def test_empty_chosen_filtered(self, tmp_path):
        raw = [{"prompt": "Question?", "chosen": "", "rejected": "Bad answer."}]
        inp, out, audit = tmp_path / "dpo_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        assert normalize_dpo_file(inp, out, audit) == 0

    def test_dpo_schema_fields(self, tmp_path):
        raw = [{"prompt": "Question?", "chosen": "Better answer.", "rejected": "Worse answer."}]
        inp, out, audit = tmp_path / "dpo_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        normalize_dpo_file(inp, out, audit)
        r = _read_jsonl(out)[0]
        for field in ["id", "source", "language", "prompt", "chosen", "rejected", "metadata"]:
            assert field in r

    def test_audit_log_written_dpo(self, tmp_path):
        raw = [{"prompt": "Q?", "chosen": "Good answer.", "rejected": "Bad answer."}]
        inp, out, audit = tmp_path / "dpo_raw.jsonl", tmp_path / "out.jsonl", tmp_path / "audit.jsonl"
        _write_jsonl(inp, raw)
        normalize_dpo_file(inp, out, audit)
        entries = _read_jsonl(audit)
        assert len(entries) == 1
        assert entries[0]["operation"] == "normalization"
