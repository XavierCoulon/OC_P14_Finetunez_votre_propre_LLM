"""
Test local du pipeline RAGAS — sans GPU, sans modèle.
Vérifie les imports, la clé Mistral et les 3 métriques sur 5 exemples médicaux.

Usage :
    uv run python scripts/10_test_ragas.py

Prérequis :
    uv add ragas langchain-mistralai sentence-transformers
    export MISTRAL_API_KEY=...
"""
import os
import sys
import types

# Patch de compatibilité : langchain-community >= 0.3 a supprimé le module vertexai
# que ragas importe en dur. On injecte un stub pour débloquer l'import.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI: pass
    _stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

# ── Exemples médicaux (instruction / reference / generated) ───────────────────
SAMPLES = [
    {
        "instruction": "What are the symptoms of myocardial infarction?",
        "reference": (
            "Symptoms of myocardial infarction include chest pain or pressure, "
            "pain radiating to the left arm or jaw, shortness of breath, nausea, "
            "sweating, and fatigue. Some patients experience atypical symptoms "
            "especially women and diabetics."
        ),
        "generated": (
            "A heart attack causes chest pain that may radiate to the arm or jaw, "
            "along with shortness of breath, cold sweats, and nausea. "
            "Fatigue and dizziness are also common."
        ),
    },
    {
        "instruction": "What is the first-line treatment for type 2 diabetes?",
        "reference": (
            "The first-line treatment for type 2 diabetes is lifestyle modification "
            "including diet and exercise, combined with metformin unless contraindicated. "
            "Metformin reduces hepatic glucose production and improves insulin sensitivity."
        ),
        "generated": (
            "Type 2 diabetes is primarily treated with metformin as the first-line drug, "
            "alongside dietary changes and physical activity. "
            "Metformin helps control blood sugar by reducing glucose production in the liver."
        ),
    },
    {
        "instruction": "How is appendicitis diagnosed?",
        "reference": (
            "Appendicitis is diagnosed based on clinical findings (rebound tenderness, "
            "pain at McBurney's point), elevated white blood cell count, and imaging "
            "such as ultrasound or CT scan. The Alvarado score helps assess probability."
        ),
        "generated": (
            "Appendicitis is typically diagnosed through physical examination showing "
            "right lower quadrant tenderness, blood tests showing elevated WBC, "
            "and confirmed by CT scan or ultrasound."
        ),
    },
    {
        "instruction": "What medications are used to treat hypertension?",
        "reference": (
            "Hypertension is treated with ACE inhibitors, ARBs, calcium channel blockers, "
            "and thiazide diuretics as first-line agents. Beta-blockers are used in "
            "specific cases. Combination therapy is often required."
        ),
        "generated": (
            "Common antihypertensive medications include ACE inhibitors like lisinopril, "
            "calcium channel blockers like amlodipine, and diuretics. "
            "The choice depends on patient comorbidities."
        ),
    },
    {
        "instruction": "What are the signs of septic shock?",
        "reference": (
            "Septic shock is characterized by persistent hypotension despite fluid "
            "resuscitation, requiring vasopressors to maintain MAP ≥65 mmHg, "
            "with serum lactate >2 mmol/L indicating tissue hypoperfusion."
        ),
        "generated": (
            "Septic shock presents with low blood pressure unresponsive to IV fluids, "
            "requiring vasopressors, elevated lactate levels, fever or hypothermia, "
            "tachycardia, and altered mental status."
        ),
    },
]


def main():
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("❌ MISTRAL_API_KEY non définie. Export : export MISTRAL_API_KEY=...")
        sys.exit(1)

    print(f"▶ Test RAGAS — {len(SAMPLES)} exemples médicaux\n")

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import FactualCorrectness, ResponseRelevancy, AnswerSimilarity
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_mistralai import ChatMistralAI
        from langchain_huggingface import HuggingFaceEmbeddings
        import ragas
        print(f"  ragas version : {ragas.__version__}")
    except ImportError as e:
        print(f"❌ Import échoué : {e}")
        print("   Installer : uv add ragas langchain-mistralai sentence-transformers")
        sys.exit(1)

    print("  Chargement MiniLM embeddings...")
    ragas_llm = LangchainLLMWrapper(
        ChatMistralAI(model="mistral-small-latest", api_key=api_key)
    )
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    ragas_samples = [
        SingleTurnSample(
            user_input=s["instruction"],
            response=s["generated"],
            reference=s["reference"],
        )
        for s in SAMPLES
    ]

    print("  Évaluation RAGAS en cours (appels Mistral)...\n")
    result = evaluate(
        dataset=EvaluationDataset(samples=ragas_samples),
        metrics=[
            FactualCorrectness(llm=ragas_llm),
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            AnswerSimilarity(embeddings=ragas_emb),
        ],
    )

    df = result.to_pandas()
    print("✅ Résultats :\n")
    print(df[["factual_correctness", "answer_relevancy", "semantic_similarity"]].to_string(index=False))
    print(f"\n  Moyennes :")
    for col in ["factual_correctness", "answer_relevancy", "semantic_similarity"]:
        print(f"    {col:30s} : {df[col].mean():.3f}")

    print("\n✅ Pipeline RAGAS opérationnel.")


if __name__ == "__main__":
    main()
