# Task 2 ViG Raw Annotation Summary

## Completeness
- CSV rows: 558
- JSON annotations: 558
- Unique image ids: 558
- All reviewed: 558/558
- needs_review: 0
- blank explanation: 27
- blank explanation with error tag: 3
- JSON missing label keys: 0
- Removed label columns: missing_objects

## Caption Quality
- correct: 239
- wrong: 179
- partial: 137
- unsure: 3

## Specificity
- specific: 326
- generic: 146
- unsure: 86

## Error Tags
- cultural_entity_missed: 98
- template_bias: 158
- object_hallucination: 127
- wrong_object_or_action: 156
- language_issue: 29

## Cultural Subsets
- Auto V_cultural reference images after boundary matching: 149
- Manual expected cultural-term images: 150
- cultural_entity_missed: auto 96/149, manual 96/150
- template_bias: auto 59/149, manual 60/150
- object_hallucination: auto 29/149, manual 28/150
- wrong_object_or_action: auto 31/149, manual 30/150
- language_issue: auto 15/149, manual 14/150

## Follow-up Flags
- Task 2 annotation is complete for ViG raw after resolving all review flags.
- Blank explanations are acceptable for clean/correct rows; check only if a later report requires every row to have a note.
- `object_hallucination` and `wrong_object_or_action` are the remaining object-level error tags; `missing_objects` was removed from the active schema.
