"""Render an artifact-only Tier B Markdown report."""
from __future__ import annotations
from typing import Any
def render_markdown(a: dict[str, Any], m: dict[str, Any]) -> str:
    meta=a["metadata"]; pct=lambda x: "n/a" if x is None else f"{x:.1%}"
    lines=["# Tier B Evaluation Report","","## Run metadata","",f"- Git SHA: `{meta['git_sha']}`",f"- Model: `{meta['model']}`",f"- Threshold: `{meta['threshold']}`",f"- Date: `{meta['timestamp']}`",f"- Persona count: {meta['persona_count']}","","## Results per category","","| Category | Count | Passed | Pass rate | Routing accuracy |","| --- | ---: | ---: | ---: | ---: |"]
    for k,v in m["results_by_category"].items(): lines.append(f"| {k} | {v['count']} | {v['passed']} | {pct(v['pass_rate'])} | {pct(v['routing_accuracy'])} |")
    i=m["injection"]; lines += ["","## Security and quality","",f"- PII leak rate: {pct(m['pii_leak_rate'])} ({m['pii_leak_count']}/{m['persona_count']})",f"- Direct injection: {pct(i['direct']['catch_rate'])} caught ({i['direct']['caught']}/{i['direct']['count']})",f"- Indirect injection: {pct(i['indirect']['catch_rate'])} caught ({i['indirect']['caught']}/{i['indirect']['count']})",f"- Routing accuracy: {pct(m['routing_accuracy'])}",f"- False approval: {m['false_approval_count']}",f"- False escalation: {m['false_escalation_count']}",f"- API/parse failures: {m['api_or_parse_failure_count']}","","## Calibration, fairness, and determinism","",f"- Brier score: {m['brier_score'] if m['brier_score'] is not None else 'n/a'}",f"- Fairness matched pairs: {len(m['fairness']['pairs'])}; all same route: {m['fairness']['all_pairs_same_route']}"]
    for p in m["fairness"]["pairs"]: lines.append(f"  - `{p['pair_id']}`: routes={', '.join(p['routes'])}; confidence range={p['confidence_range']}")
    lines += [f"- Determinism: {m['determinism']['status']} — {m['determinism']['finding']}","","## Known limitations/findings","","- Metrics above were computed from the saved JSON; they do not re-call Gemini.","- A one-run artifact cannot establish determinism; compare repeated immutable artifacts.",""]
    return "\n".join(lines)
