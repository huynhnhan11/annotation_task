# Cultural Branch Evaluation Todo / Plan

Updated: 2026-06-07

## Done
- [x] Task 0 train-only `V_cultural`: finalized active vocab at `task/output/task0/V_cultural_train_only.json` (23 terms).
- [x] Task 1 ViG raw cultural entity recall probe: active output at `task/output/task1/task1_cultural_entity_recall.json`.
- [x] Task 1 aggregate: ViG raw matched 65/180 cultural concept-image references, recall 0.3611.
- [x] Task 2 ViG raw manual annotation: active outputs at `task/output/task2/vig_raw_annotations.json` and `.csv`.
- [x] Task 2 cleanup: 558/558 reviewed, needs_review 0, removed legacy object-missing label from schema.
- [x] Task 2 audit/summary regenerated: `task/output/task2/task2_audit_report.json`, `task/output/task2/task2_summary.md`.
- [x] Annotation app moved under `task/apps/annotation_app`.
- [x] Task 5 ViG raw LCR diagnostic: active output at `task/output/task5/lcr_lor_metrics.json`; 65/180, LCR 0.3611.

## Next
- [ ] Task 3 paired comparison: use `task/output/task2/task3_candidate_cases_from_task2.csv` as the reviewed case pool and write a concise qualitative side-by-side report.
- [ ] Task 2 paper table/delta: compare ViG raw counts against the baseline A/C counts if those baseline annotation files are available.
- [ ] Inter-annotator agreement: Cohen's kappa is still pending unless a second independent annotator file exists for Task 0/Task 2.
- [ ] Task 4 mechanism analysis: extract 16 LCQM slot top phrases and eta_loc distribution from the trained ViG checkpoint/model internals.

## Current Task 2 Counts
- caption_quality: correct=239, wrong=179, partial=137, unsure=3
- specificity: specific=326, generic=146, unsure=86
- error_tags: cultural_entity_missed=98, template_bias=158, object_hallucination=127, wrong_object_or_action=156, language_issue=29
