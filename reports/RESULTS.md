# SpecDecodeLab — measured local run

Pretrained SmolLM2 target, fitted token-bigram draft, greedy verification only. No sampling or KV-cache optimization; negative speed results retained.

## Results

```json
{
  "accepted_draft_tokens": 2,
  "proposed_tokens": 32,
  "target_calls": 11,
  "greedy_seconds": 1.4648636000019906,
  "speculative_seconds": 1.418245550001302,
  "checks": [
    "Speculative verification exactly matches greedy target output"
  ],
  "scope": "Pretrained SmolLM2 target, fitted token-bigram draft, greedy verification only. No sampling or KV-cache optimization; negative speed results retained."
}
```

Local experiment assertions passed. This does not establish production readiness or performance on real customer data. Source fingerprints and dependency versions are in results.json.
