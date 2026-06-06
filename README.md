# Cultural Branch Evaluation Workspace

This folder contains the ViG cultural-branch evaluation workflow. Everything
specific to this evaluation now lives under `task/`.

## Directory Layout

```text
task/
├── apps/
│   └── annotation_app/              # Local ViG raw annotation web app
├── data/
│   └── predictions/                 # ViG prediction inputs used by scripts/app
├── docs/                            # Annotation and evaluation instructions
├── scripts/
│   ├── task0/                       # Vocabulary mining / clustering
│   ├── task1/                       # Cultural entity recall probe
│   ├── task2/                       # Annotation sheet / draft helpers
│   └── task5/                       # LCR diagnostic metrics
└── output/
    ├── task0/                       # V_cultural and candidate outputs
    ├── task1/                       # Recall outputs
    ├── task2/                       # Manual annotations, audit, Task 3 case pool
    ├── task5/                       # LCR outputs
    ├── reports/                     # Cross-task todo/progress reports
    └── _archive/                    # Legacy outputs kept for traceability
```

## Active Inputs

- `task/data/predictions/vig_raw_epoch10_details.json`
- `data/test_data.json`
- `data/public-test-images/`

## Active Outputs

- `task/output/task0/V_cultural_train_only.json`
- `task/output/task1/task1_cultural_entity_recall.json`
- `task/output/task2/vig_raw_annotations.csv`
- `task/output/task2/vig_raw_annotations.json`
- `task/output/task2/task2_summary.md`
- `task/output/task2/task2_audit_report.json`
- `task/output/task2/task3_candidate_cases_from_task2.csv`
- `task/output/task5/lcr_lor_metrics.json`
- `task/output/reports/cultural_branch_todo_plan.md`

## Run Commands

```bash
# Annotation app
python3 task/apps/annotation_app/server.py --host 127.0.0.1 --port 8765

# Task 0
python3 task/scripts/task0/mine_candidates_train_only.py
python3 task/scripts/task0/cluster_candidates_train_only.py

# Task 1
python3 task/scripts/task1/cultural_entity_recall.py

# Task 2 helpers
python3 task/scripts/task2/prepare_vig_raw_annotation.py
python3 task/scripts/task2/autolabel_vig_raw_draft.py

# Task 5
python3 task/scripts/task5/lcr_lor_metrics.py
```

## Current Status

- Task 0 train-only vocabulary is active.
- Task 1 ViG raw recall has been computed.
- Task 2 ViG raw annotation is complete: 558/558 reviewed, `needs_review=0`.
- Task 5 ViG raw LCR diagnostic has been computed.
- Next main work item is Task 3 paired qualitative comparison.
