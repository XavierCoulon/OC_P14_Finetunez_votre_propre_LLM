"""Tests du pipeline d'anonymisation (fonctions pures — sans charger Presidio/spaCy)."""
from unittest.mock import MagicMock

from src.data_pipeline.anonymizer import (
    MEDICAL_ALLOWLIST,
    NRP_SCORE_THRESHOLD,
    _apply_name_regex,
    _filter_results,
)


def _mock_result(entity_type: str, start: int, end: int, score: float):
    r = MagicMock()
    r.entity_type = entity_type
    r.start = start
    r.end = end
    r.score = score
    return r


# ── _filter_results ───────────────────────────────────────────────────────────

class TestFilterResults:

    def test_short_entity_filtered(self):
        text = "Hi AB dolor sit."
        result = _mock_result("PERSON", 3, 5, 0.9)  # "AB" = 2 chars
        assert _filter_results([result], text) == []

    def test_long_entity_kept(self):
        text = "Patient John came in today."
        result = _mock_result("PERSON", 8, 12, 0.9)  # "John" = 4 chars
        assert len(_filter_results([result], text)) == 1

    def test_medical_allowlist_filtered(self):
        term = "sjogren"
        text = f"Diagnosed with {term} syndrome."
        start = text.lower().index(term)
        result = _mock_result("PERSON", start, start + len(term), 0.9)
        assert _filter_results([result], text) == []

    def test_nrp_low_score_filtered(self):
        text = "Referred to WHO guidelines today."
        result = _mock_result("NRP", 12, 15, NRP_SCORE_THRESHOLD - 0.1)
        assert _filter_results([result], text) == []

    def test_nrp_high_score_kept(self):
        text = "The NRP group published guidelines."
        result = _mock_result("NRP", 4, 7, NRP_SCORE_THRESHOLD + 0.05)
        assert len(_filter_results([result], text)) == 1

    def test_multiple_entities_partial_filter(self):
        text = "Patient John has AB condition today."
        r1 = _mock_result("PERSON", 8, 12, 0.9)   # "John" → kept
        r2 = _mock_result("PERSON", 18, 20, 0.9)   # "AB" → filtered
        assert len(_filter_results([r1, r2], text)) == 1

    def test_empty_results(self):
        assert _filter_results([], "any text") == []

    def test_all_medical_allowlist_terms_are_filtered(self):
        """Vérifie que chaque terme de l'allowlist est bien filtré."""
        for term in list(MEDICAL_ALLOWLIST)[:5]:  # échantillon de 5
            text = f"Condition called {term} is present."
            start = text.lower().index(term)
            result = _mock_result("PERSON", start, start + len(term), 0.95)
            assert _filter_results([result], text) == [], f"{term} should be filtered"


# ── _apply_name_regex ─────────────────────────────────────────────────────────

class TestApplyNameRegex:

    def test_my_name_is_replaced(self):
        text = "my name is Amber and I need help."
        result, count = _apply_name_regex(text)
        assert "<PERSON>" in result
        assert "Amber" not in result
        assert count == 1

    def test_dear_replaced(self):
        result, count = _apply_name_regex("Dear John, please find attached.")
        assert "<PERSON>" in result
        assert count == 1

    def test_hi_replaced(self):
        result, count = _apply_name_regex("Hi Sarah, how are you feeling?")
        assert "<PERSON>" in result
        assert count == 1

    def test_thanks_replaced(self):
        result, count = _apply_name_regex("Thanks, Michael for your help today.")
        assert "<PERSON>" in result
        assert count == 1

    def test_excluded_words_not_replaced(self):
        result, count = _apply_name_regex("Dear Doctor, please advise.")
        assert "Doctor" in result
        assert count == 0

    def test_no_match_returns_unchanged(self):
        text = "The patient presents with chest pain."
        result, count = _apply_name_regex(text)
        assert result == text
        assert count == 0

    def test_multiple_patterns(self):
        # "Thanks" doit être capitalisé pour matcher le pattern \bThanks[,]?
        text = "Hi Alice, my name is Robert, Thanks, Carol."
        result, count = _apply_name_regex(text)
        assert count == 3
        assert "Alice" not in result
        assert "Robert" not in result
        assert "Carol" not in result

    def test_empty_string(self):
        result, count = _apply_name_regex("")
        assert result == ""
        assert count == 0

    def test_lowercase_name_not_matched(self):
        # Les patterns cherchent [A-Z][a-z]{2,} — un prénom tout minuscule ne doit pas matcher
        result, count = _apply_name_regex("my name is amber.")
        assert count == 0
