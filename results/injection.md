## Prompt-injection detection

Threshold 0.5, 697 prompts. P = precision, R = recall, FPR = false-positive rate on benign prompts.

| Dataset | Detector | N | P | R | F1 | FPR |
|---|---|---|---|---|---|---|
| clinical (ours) | heuristic | 181 | 100.0% | 60.9% | 75.7% | 0.0% |
| clinical (ours) | classifier | 181 | 71.2% | 53.6% | 61.2% | 13.4% |
| clinical (ours) | ensemble | 181 | 77.6% | 75.4% | 76.5% | 13.4% |
| clinical: benign_note | heuristic | 40 | – | – | – | 0.0% |
| clinical: benign_note | classifier | 40 | – | – | – | 0.0% |
| clinical: benign_note | ensemble | 40 | – | – | – | 0.0% |
| clinical: benign_trigger | heuristic | 72 | – | – | – | 0.0% |
| clinical: benign_trigger | classifier | 72 | 0.0% | – | – | 20.8% |
| clinical: benign_trigger | ensemble | 72 | 0.0% | – | – | 20.8% |
| clinical: direct_attack | heuristic | 29 | 100.0% | 58.6% | 73.9% | – |
| clinical: direct_attack | classifier | 29 | 100.0% | 72.4% | 84.0% | – |
| clinical: direct_attack | ensemble | 29 | 100.0% | 79.3% | 88.5% | – |
| clinical: indirect_note | heuristic | 40 | 100.0% | 62.5% | 76.9% | – |
| clinical: indirect_note | classifier | 40 | 100.0% | 40.0% | 57.1% | – |
| clinical: indirect_note | ensemble | 40 | 100.0% | 72.5% | 84.1% | – |
| deepset/prompt-injections | heuristic | 116 | 100.0% | 8.3% | 15.4% | 0.0% |
| deepset/prompt-injections | classifier | 116 | 100.0% | 36.7% | 53.7% | 0.0% |
| deepset/prompt-injections | ensemble | 116 | 100.0% | 38.3% | 55.4% | 0.0% |
| jackhhao/jailbreak-classification | heuristic | 400 | 100.0% | 29.8% | 45.9% | 0.0% |
| jackhhao/jailbreak-classification | classifier | 400 | 95.9% | 83.0% | 89.0% | 1.9% |
| jackhhao/jailbreak-classification | ensemble | 400 | 96.0% | 84.4% | 89.8% | 1.9% |

Latency per prompt (CPU):

| Detector | p50 | p95 |
|---|---|---|
| heuristic | 0.16 ms | 1.98 ms |
| classifier | 254.72 ms | 2057.77 ms |
| ensemble | 254.94 ms | 2060.04 ms |
