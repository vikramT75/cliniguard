## PHI detection

300 synthetic notes, 2135 gold PHI spans. *Full recall* counts an identifier as protected only if every character is masked; *note leak rate* is the share of notes where at least one identifier survives.

| Config | Recall (full) | Recall (any overlap) | Precision | Note leak rate | p50 latency |
|---|---|---|---|---|---|
| Presidio default (threshold 0) | 85.0% | 94.8% | 69.3% | 67.0% | 37.6 ms |
| Presidio default (threshold 0.5) | 72.7% | 80.1% | 65.7% | 81.7% | 42.86 ms |
| Clinical recognizers (ours, 0.5) | 97.4% | 99.2% | 100.0% | 17.7% | 36.96 ms |

Recall by identifier type:

| Type | Presidio default (threshold 0) | Presidio default (threshold 0.5) | Clinical recognizers (ours, 0.5) |
|---|---|---|---|
| ADDRESS | 18.1% | 18.1% | 94.0% |
| DATE | 81.1% | 81.1% | 100.0% |
| DATE_OF_BIRTH | 95.2% | 95.2% | 100.0% |
| DEA | 100.0% | 100.0% | 100.0% |
| EMAIL | 100.0% | 100.0% | 100.0% |
| LOCATION | 90.2% | 90.2% | 98.5% |
| MEMBER_ID | 10.9% | 10.9% | 100.0% |
| MRN | 100.0% | 66.4% | 100.0% |
| NPI | 100.0% | 83.0% | 100.0% |
| PERSON | 95.6% | 95.6% | 94.0% |
| PHONE | 91.5% | 6.4% | 100.0% |
| SSN | 100.0% | 100.0% | 100.0% |
| ZIP | 55.6% | 55.6% | 92.5% |
