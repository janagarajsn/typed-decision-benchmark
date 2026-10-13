# Golden datasets

Fixed evaluation sets with known-correct answers. The benchmark runs every model on the same items, so results are comparable across runs, models and code changes.

| File | Task | Questions | Items |
|------|------|-----------|-------|
| `support_triage.jsonl` | Support ticket triage: synthetic emails built from phrase banks, so labels are correct by construction | `intent` choice, `is_urgent` / `refund_requested` / `churn_risk` noul, `frustration` / `impact_scope` score | 100 |
| `spam_detection.jsonl` | Spam and phishing detection | `is_spam`, `is_phishing` (noul) | 100 |
| `severity_scoring.jsonl` | Bug-report severity on a 0–3 rubric | `severity` (score) | 100 |
| `topic_routing.jsonl` | Route a smart-home company's email to a team | `department` (choice, 7 teams) | 100 |
| `policy_decisions.jsonl` | Apply a written refund policy to a case | `decision`, `reason` (choice) | 100 |

## Format

One JSON object per line. Line 1 is the header; every other line is an item. The full spec and the validator are in [src/golden.py](../../src/golden.py).

```json
{"meta": {"name": "spam_detection", "title": "...", "description": "...", "context": {}, "questions": {...}, "tags": {...}}}
{"id": "spam-001", "input": {"email": "..."}, "labels": {"is_spam": true, "is_phishing": true},
 "tags": ["phishing_credential", "hard"], "rationale": "...", "reviewed": false}
```

- `input` is the state sent to each model. The header's optional `context` is merged into it (the policy dataset uses this so every case is sent with the policy).
- Label values: choice → option key, noul → `true`/`false`, score → level index (0 = first rubric level).
- `tags` drive the "Accuracy by tag" table. Every dataset tags each item `easy` or `hard`, plus scenario tags declared in the header.
- `rationale` is one sentence explaining the answer, for reviewers.

## Reviewing

Every item in all five datasets has been reviewed by a person and is marked `"reviewed": true`. When you add or change an item, set `"reviewed": false`. The dashboard then shows "N/M reviewed" next to that dataset until every item is checked again. To review an item:

1. Read `input` and decide the answer yourself before looking at `labels`.
2. If you agree, set `"reviewed": true`.
3. If you disagree, fix `labels` and `rationale`. If a careful reader could reasonably go either way, **delete the item**: a golden set should only contain items with one defensible answer.

`policy_decisions` labels were computed by a script that applies the policy's rules in order, so review there is mainly about whether each case states its facts clearly.

After editing, validate from the project root:

```bash
python src/golden.py
```

## Adding a dataset

Drop a new `<name>.jsonl` here with a header whose `name` matches the file name. It shows up in the dashboard's picker automatically; no code changes are needed.
