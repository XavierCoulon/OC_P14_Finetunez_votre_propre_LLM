"""
Benchmark de latence de l'API CHSA Triage.

Usage:
    uv run python scripts/benchmark_latency.py [--url http://localhost:8080] [--key YOUR_KEY]
"""
import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import httpx

CASES = [
    "Patient : femme 34 ans, fièvre 39.5°C depuis 48h, toux sèche, dyspnée légère.",
    "Homme 67 ans, douleur thoracique irradiant bras gauche, sueurs, antécédent IDM.",
    "Enfant 8 ans, convulsions tonico-cloniques 3 min, retour à conscience, T° 38.8°C.",
    "Femme 52 ans, céphalée brutale 10/10, raideur nuque, photophobie.",
    "Homme 45 ans, douleur abdominale diffuse, défense généralisée, fièvre 38.2°C.",
]

EDGE_CASES = [
    ("vide", ""),
    ("très_long", "Symptôme. " * 200),
    ("spéciaux", "Patient: <script>alert('x')</script> & douleur 'thoracique'"),
]


async def call_triage(client: httpx.AsyncClient, url: str, key: str, text: str) -> float:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{url}/v1/triage",
            json={"patient_description": text, "think": False},
            headers={"X-API-Key": key},
            timeout=60.0,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERREUR: {e}")
        return -1.0
    return (time.monotonic() - t0) * 1000


async def run_benchmark(url: str, key: str) -> dict:
    results = {}

    async with httpx.AsyncClient() as client:
        # 1. Latence nominale — 10 requêtes séquentielles
        print("\n[1/3] Latence nominale (10 requêtes séquentielles)...")
        lats = []
        for i in range(10):
            text = CASES[i % len(CASES)]
            ms = await call_triage(client, url, key, text)
            if ms >= 0:
                lats.append(ms)
                print(f"  #{i+1}: {ms:.0f} ms")

        if lats:
            results["nominal"] = {
                "n": len(lats),
                "p50_ms": round(statistics.median(lats), 1),
                "p95_ms": round(sorted(lats)[int(len(lats) * 0.95)], 1) if len(lats) >= 5 else None,
                "min_ms": round(min(lats), 1),
                "max_ms": round(max(lats), 1),
            }

        # 2. Charge concurrente — 5 requêtes simultanées
        print("\n[2/3] Charge concurrente (5 requêtes simultanées)...")
        t0 = time.monotonic()
        tasks = [call_triage(client, url, key, CASES[i % len(CASES)]) for i in range(5)]
        concurrent_lats = await asyncio.gather(*tasks)
        wall_ms = (time.monotonic() - t0) * 1000
        valid = [l for l in concurrent_lats if l >= 0]
        results["concurrent"] = {
            "n": 5,
            "wall_ms": round(wall_ms, 1),
            "individual_ms": [round(l, 1) for l in concurrent_lats],
        }
        print(f"  Wall time: {wall_ms:.0f} ms")

        # 3. Cas limites
        print("\n[3/3] Cas limites...")
        edge_results = {}
        for name, text in EDGE_CASES:
            if not text:
                # prompt vide → doit retourner 422
                try:
                    resp = await client.post(
                        f"{url}/v1/triage",
                        json={"patient_description": text, "think": False},
                        headers={"X-API-Key": key},
                        timeout=10.0,
                    )
                    edge_results[name] = {"status": resp.status_code}
                    print(f"  {name}: status {resp.status_code}")
                except Exception as e:
                    edge_results[name] = {"error": str(e)}
            else:
                ms = await call_triage(client, url, key, text[:4000])
                edge_results[name] = {"latency_ms": round(ms, 1) if ms >= 0 else "error"}
                print(f"  {name}: {ms:.0f} ms" if ms >= 0 else f"  {name}: ERREUR")
        results["edge_cases"] = edge_results

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--key", default="")
    args = parser.parse_args()

    print(f"Benchmark CHSA Triage API → {args.url}")
    results = asyncio.run(run_benchmark(args.url, args.key))

    print("\n=== RÉSULTATS ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out = Path("audit/benchmark_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nRésultats sauvegardés dans {out}")

    # Affiche le tableau markdown pour le rapport
    nom = results.get("nominal", {})
    if nom:
        print("\n### Métriques de latence (nominale)")
        print(f"| P50 | P95 | Min | Max |")
        print(f"|-----|-----|-----|-----|")
        print(f"| {nom.get('p50_ms')} ms | {nom.get('p95_ms')} ms | {nom.get('min_ms')} ms | {nom.get('max_ms')} ms |")


if __name__ == "__main__":
    main()
