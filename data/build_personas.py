from __future__ import annotations

import json
import random
from pathlib import Path

from generators import (
    gen_clean_approve,
    gen_clean_deny,
    gen_inconsistency_obvious,
    gen_inconsistency_inferential,
    gen_looks_like_conflict,
    gen_ambiguous_escalate,
    gen_injection_direct,
    gen_injection_indirect,
    gen_pii_obfuscated_adversarial,
    gen_degraded_document,
    gen_fairness_pairs,
)

from render_pdf import render_persona

SEED = 42


def main():
    rng = random.Random(SEED)

    personas = []

    # Counts from func_testing_synthetic.md
    personas.extend(gen_clean_approve(rng, 25))
    personas.extend(gen_clean_deny(rng, 10))
    personas.extend(gen_inconsistency_obvious(rng, 15))
    personas.extend(gen_inconsistency_inferential(rng, 10))
    personas.extend(gen_looks_like_conflict(rng, 10))
    personas.extend(gen_ambiguous_escalate(rng, 10))
    personas.extend(gen_injection_direct(rng, 10))
    personas.extend(gen_injection_indirect(rng, 8))
    personas.extend(gen_pii_obfuscated_adversarial(rng, 15))
    personas.extend(gen_degraded_document(rng, 10))
    personas.extend(gen_fairness_pairs(rng, 10))   # 10 pairs = 20 personas

    out_dir = Path("generated_personas")
    pdf_dir = out_dir / "pdfs"

    out_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)

    for persona in personas:
        render_persona(persona, str(pdf_dir))

    with open(out_dir / "personas.json", "w", encoding="utf-8") as f:
        json.dump(
            [p.model_dump(mode="json") for p in personas],
            f,
            indent=2,
        )

    print(f"Generated {len(personas)} personas.")
    print(f"JSON saved to {out_dir/'personas.json'}")
    print(f"PDFs saved to {pdf_dir}")


if __name__ == "__main__":
    main()