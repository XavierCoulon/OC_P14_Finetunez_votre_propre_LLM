"""Tests du validator SFT et DPO."""
import json
from pathlib import Path

from src.data_pipeline.validator import validate_dpo_record, validate_file, validate_sft_record

VALID_SFT = {
    "id": "sft_medquad_000001",
    "source": "medquad",
    "language": "en",
    "instruction": "What are the symptoms of diabetes?",
    "response": "Diabetes symptoms include increased thirst, frequent urination, and fatigue.",
    "metadata": {},
}

VALID_DPO = {
    "id": "dpo_ultramedical_000001",
    "source": "ultramedical_preference",
    "prompt": "What is the first-line treatment for hypertension?",
    "chosen": {"role": "assistant", "content": "ACE inhibitors are first-line for hypertension."},
    "rejected": {"role": "assistant", "content": "No treatment is needed."},
    "metadata": {},
}


# ── SFT record ────────────────────────────────────────────────────────────────

class TestValidateSftRecord:

    def test_valid_record_no_errors(self):
        assert validate_sft_record(VALID_SFT) == []

    def test_missing_instruction(self):
        errors = validate_sft_record({**VALID_SFT, "instruction": ""})
        assert any("manquant" in e or "courte" in e for e in errors)

    def test_missing_response(self):
        errors = validate_sft_record({**VALID_SFT, "response": ""})
        assert any("manquant" in e or "courte" in e for e in errors)

    def test_response_too_short(self):
        errors = validate_sft_record({**VALID_SFT, "response": "Yes."})
        assert any("response trop courte" in e for e in errors)

    def test_instruction_too_long(self):
        errors = validate_sft_record({**VALID_SFT, "instruction": " ".join(["w"] * 2001)})
        assert any("trop longue" in e for e in errors)

    def test_pii_email_detected(self):
        errors = validate_sft_record({**VALID_SFT, "instruction": "Email me at john@example.com please."})
        assert any("PII" in e for e in errors)

    def test_pii_phone_detected(self):
        errors = validate_sft_record({**VALID_SFT, "response": "Call 0612345678 for an appointment today here."})
        assert any("PII" in e for e in errors)

    def test_pii_long_number_detected(self):
        errors = validate_sft_record({**VALID_SFT, "instruction": "My SS number is 1234567890123 and I need help."})
        assert any("PII" in e for e in errors)

    def test_missing_id(self):
        errors = validate_sft_record({**VALID_SFT, "id": ""})
        assert any("id" in e for e in errors)

    def test_missing_source(self):
        errors = validate_sft_record({**VALID_SFT, "source": ""})
        assert any("source" in e for e in errors)


# ── DPO record ────────────────────────────────────────────────────────────────

class TestValidateDpoRecord:

    def test_valid_record_no_errors(self):
        assert validate_dpo_record(VALID_DPO) == []

    def test_empty_prompt(self):
        errors = validate_dpo_record({**VALID_DPO, "prompt": ""})
        assert any("prompt" in e for e in errors)

    def test_chosen_equals_rejected(self):
        content = "Same answer."
        r = {**VALID_DPO, "chosen": {"role": "assistant", "content": content},
             "rejected": {"role": "assistant", "content": content}}
        errors = validate_dpo_record(r)
        assert any("chosen == rejected" in e for e in errors)

    def test_missing_chosen_content(self):
        errors = validate_dpo_record({**VALID_DPO, "chosen": {"role": "assistant", "content": ""}})
        assert any("chosen" in e for e in errors)

    def test_missing_rejected_content(self):
        errors = validate_dpo_record({**VALID_DPO, "rejected": {"role": "assistant", "content": ""}})
        assert any("rejected" in e for e in errors)

    def test_missing_id(self):
        errors = validate_dpo_record({**VALID_DPO, "id": ""})
        assert any("id" in e for e in errors)


# ── validate_file ──────────────────────────────────────────────────────────────

class TestValidateFile:

    def _write(self, path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_all_valid_sft(self, tmp_path):
        records = [{**VALID_SFT, "id": f"sft_{i}"} for i in range(3)]
        p = tmp_path / "test.jsonl"
        self._write(p, records)
        result = validate_file(p, "sft")
        assert result["total"] == 3
        assert result["valid"] == 3
        assert result["errors"] == {}

    def test_partial_errors_sft(self, tmp_path):
        records = [VALID_SFT, {**VALID_SFT, "id": "bad", "response": "Too short."}]
        p = tmp_path / "test.jsonl"
        self._write(p, records)
        result = validate_file(p, "sft")
        assert result["total"] == 2
        assert result["valid"] == 1
        assert len(result["errors"]) == 1

    def test_all_valid_dpo(self, tmp_path):
        records = [{**VALID_DPO, "id": f"dpo_{i}"} for i in range(2)]
        p = tmp_path / "dpo.jsonl"
        self._write(p, records)
        result = validate_file(p, "dpo")
        assert result["total"] == 2
        assert result["valid"] == 2

    def test_error_count_matches(self, tmp_path):
        records = [
            {**VALID_SFT, "id": "ok"},
            {**VALID_SFT, "id": "bad1", "response": "Short."},
            {**VALID_SFT, "id": "bad2", "instruction": ""},
        ]
        p = tmp_path / "test.jsonl"
        self._write(p, records)
        result = validate_file(p, "sft")
        assert result["total"] == 3
        assert result["valid"] == 1
        assert len(result["errors"]) == 2
