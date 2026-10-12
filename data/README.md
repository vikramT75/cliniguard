# Data

Everything here is either public-domain, permissively licensed, or synthetic. No real patient
data is used anywhere in this project.

| File | What it is | Source / licence |
|---|---|---|
| `openfda_labels.json` | Sections of 30 FDA drug labels (dosage, contraindications, warnings, indications…) used as grounding context | [openFDA drug label API](https://open.fda.gov/apis/drug/label/), public domain. Refresh with `uv run guardrail-data --refresh-corpus` |
| `eval/injection_public.jsonl` | 116 prompts from `deepset/prompt-injections` (test split) + 400 from `jackhhao/jailbreak-classification` (test split) | Hugging Face, both Apache-2.0 |
| `eval/injection_clinical.jsonl` | Our clinical set: benign questions containing trigger words, benign notes, direct attacks, injections hidden in notes, plus a held-out set written after the regex rules were frozen | Written for this project (templated, seeded) |
| `eval/phi_notes.jsonl` | 300 synthetic clinical notes (progress, discharge, referral, pharmacy, portal) with character-exact PHI spans | Generated with Faker (seed 7); names/IDs are fake, NPIs and DEA numbers have valid checksums |
| `eval/hallucination.jsonl` | 190 (context, answer) pairs on openFDA passages: 90 faithful extractive answers and 100 with one injected error (20 each: dose, frequency, drug swap, negation, fabricated claim) | Generated from the openFDA corpus (seed 7) |

`audit.db` is created at runtime by the proxy and is git-ignored.

Rebuild everything: `uv run guardrail-data` (add `--skip-public` to avoid the Hugging Face download).
