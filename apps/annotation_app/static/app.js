const state = {
  items: [],
  filtered: [],
  currentIndex: 0,
  selected: null,
  outputs: {},
  progress: { saved: 0, reviewed: 0, needs_review: 0 },
  segmentValues: {
    caption_quality: "",
    specificity: "",
  },
};

const els = {
  progressText: document.getElementById("progressText"),
  refreshBtn: document.getElementById("refreshBtn"),
  searchInput: document.getElementById("searchInput"),
  statusFilter: document.getElementById("statusFilter"),
  itemList: document.getElementById("itemList"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  imageId: document.getElementById("imageId"),
  imageMeta: document.getElementById("imageMeta"),
  mainImage: document.getElementById("mainImage"),
  predictionText: document.getElementById("predictionText"),
  referenceTerms: document.getElementById("referenceTerms"),
  predictionTerms: document.getElementById("predictionTerms"),
  draftMetrics: document.getElementById("draftMetrics"),
  draftLabels: document.getElementById("draftLabels"),
  draftExplanation: document.getElementById("draftExplanation"),
  applyDraftBtn: document.getElementById("applyDraftBtn"),
  referenceList: document.getElementById("referenceList"),
  saveState: document.getElementById("saveState"),
  annotatorInput: document.getElementById("annotatorInput"),
  form: document.getElementById("annotationForm"),
  reviewedInput: document.getElementById("reviewedInput"),
  needsReviewInput: document.getElementById("needsReviewInput"),
  culturalMissedInput: document.getElementById("culturalMissedInput"),
  templateBiasInput: document.getElementById("templateBiasInput"),
  hallucinationInput: document.getElementById("hallucinationInput"),
  wrongObjectInput: document.getElementById("wrongObjectInput"),
  languageIssueInput: document.getElementById("languageIssueInput"),
  expectedTermsInput: document.getElementById("expectedTermsInput"),
  predictedTermsInput: document.getElementById("predictedTermsInput"),
  explanationInput: document.getElementById("explanationInput"),
  jsonLink: document.getElementById("jsonLink"),
  csvLink: document.getElementById("csvLink"),
};

function termNames(terms) {
  return (terms || []).map((item) => item.term);
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function termsToInput(terms) {
  return unique(terms).join(", ");
}

function inputToTerms(value) {
  return value
    .split(/[,;]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function textForSearch(item) {
  return [
    item.image_id,
    item.prediction,
    ...(item.references || []),
    ...termNames(item.reference_terms),
    ...termNames(item.prediction_terms),
  ]
    .join(" ")
    .toLowerCase();
}

function annotationFor(item) {
  return item?.annotation || null;
}

function labelsFor(item) {
  return annotationFor(item)?.labels || {};
}

function updateProgress() {
  const total = state.items.length;
  const { saved, reviewed, needs_review: needsReview } = state.progress;
  els.progressText.textContent = `${reviewed}/${total} reviewed · ${saved} saved · ${needsReview} review`;
}

function renderChipContainer(node, terms) {
  node.innerHTML = "";
  const list = terms || [];
  if (!list.length) {
    const chip = document.createElement("span");
    chip.className = "chip muted";
    chip.textContent = "none";
    node.appendChild(chip);
    return;
  }
  list.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${item.term}${item.facet ? ` · ${item.facet}` : ""}`;
    node.appendChild(chip);
  });
}

function renderDraft(item) {
  const draft = item?.ai_draft || {};
  const labels = draft.labels || {};
  els.draftMetrics.textContent = `overlap ${draft.reference_overlap ?? "n/a"} · source ${draft.source || "draft"}`;
  els.draftExplanation.textContent = draft.explanation || "Không có draft.";
  els.draftLabels.innerHTML = "";

  const labelItems = [
    ["quality", labels.caption_quality],
    ["specificity", labels.specificity],
    ["cultural_missed", labels.cultural_entity_missed],
    ["template_bias", labels.template_bias],
    ["language_issue", labels.language_issue],
    ["needs_review", labels.needs_review],
  ];

  labelItems.forEach(([name, value]) => {
    if (value === "" || value === undefined || value === false) return;
    const node = document.createElement("span");
    node.className = "draft-label";
    if (value === true && (name === "cultural_missed" || name === "template_bias" || name === "language_issue")) {
      node.classList.add("negative");
    }
    if (name === "quality" && value === "correct") {
      node.classList.add("positive");
    }
    node.textContent = value === true ? name : `${name}: ${value}`;
    els.draftLabels.appendChild(node);
  });

  if (!els.draftLabels.children.length) {
    const node = document.createElement("span");
    node.className = "draft-label positive";
    node.textContent = "no flagged draft labels";
    els.draftLabels.appendChild(node);
  }
}

function setSegment(field, value) {
  state.segmentValues[field] = value || "";
  document.querySelectorAll(`.segmented[data-field="${field}"] button`).forEach((button) => {
    button.classList.toggle("active", button.dataset.value === state.segmentValues[field]);
  });
}

function resetForm(item) {
  const ann = annotationFor(item);
  const labels = labelsFor(item);
  const defaultExpected = termNames(item.reference_terms);
  const defaultPredicted = termNames(item.prediction_terms);

  els.reviewedInput.checked = Boolean(labels.reviewed);
  els.needsReviewInput.checked = Boolean(labels.needs_review);
  els.culturalMissedInput.checked = Boolean(labels.cultural_entity_missed);
  els.templateBiasInput.checked = Boolean(labels.template_bias);
  els.hallucinationInput.checked = Boolean(labels.object_hallucination);
  els.wrongObjectInput.checked = Boolean(labels.wrong_object_or_action);
  els.languageIssueInput.checked = Boolean(labels.language_issue);
  setSegment("caption_quality", labels.caption_quality || "");
  setSegment("specificity", labels.specificity || "");

  els.expectedTermsInput.value = termsToInput(ann?.expected_cultural_terms || defaultExpected);
  els.predictedTermsInput.value = termsToInput(ann?.predicted_cultural_terms || defaultPredicted);
  els.explanationInput.value = ann?.explanation || "";
  els.saveState.textContent = ann?.updated_at ? `Saved ${ann.updated_at}` : "Not saved";
}

function applyDraftToForm() {
  const item = state.selected;
  const draft = item?.ai_draft;
  if (!draft) return;
  const labels = draft.labels || {};

  els.reviewedInput.checked = false;
  els.needsReviewInput.checked = Boolean(labels.needs_review);
  els.culturalMissedInput.checked = Boolean(labels.cultural_entity_missed);
  els.templateBiasInput.checked = Boolean(labels.template_bias);
  els.hallucinationInput.checked = Boolean(labels.object_hallucination);
  els.wrongObjectInput.checked = Boolean(labels.wrong_object_or_action);
  els.languageIssueInput.checked = Boolean(labels.language_issue);
  setSegment("caption_quality", labels.caption_quality || "");
  setSegment("specificity", labels.specificity || "");
  els.expectedTermsInput.value = termsToInput(draft.expected_cultural_terms || []);
  els.predictedTermsInput.value = termsToInput(draft.predicted_cultural_terms || []);
  els.explanationInput.value = draft.explanation || "";
  els.saveState.textContent = "Draft applied · review then save";
}

function renderCurrent() {
  const item = state.filtered[state.currentIndex];
  state.selected = item || null;

  if (!item) {
    els.imageId.textContent = "No item";
    els.imageMeta.textContent = "";
    els.mainImage.removeAttribute("src");
    els.predictionText.textContent = "";
    els.referenceList.innerHTML = "";
    renderChipContainer(els.referenceTerms, []);
    renderChipContainer(els.predictionTerms, []);
    return;
  }

  els.imageId.textContent = `#${item.image_id}`;
  els.imageMeta.textContent = item.image_filename;
  els.mainImage.src = item.image_url;
  els.predictionText.textContent = item.prediction || "";

  renderChipContainer(els.referenceTerms, item.reference_terms);
  renderChipContainer(els.predictionTerms, item.prediction_terms);
  renderDraft(item);

  els.referenceList.innerHTML = "";
  (item.references || []).forEach((reference) => {
    const li = document.createElement("li");
    li.textContent = reference;
    els.referenceList.appendChild(li);
  });

  resetForm(item);
  renderList();
}

function itemMatchesFilter(item) {
  const query = els.searchInput.value.trim().toLowerCase();
  const status = els.statusFilter.value;
  const labels = labelsFor(item);

  if (query && !textForSearch(item).includes(query)) {
    return false;
  }

  if (status === "reviewed") return Boolean(labels.reviewed);
  if (status === "unreviewed") return !labels.reviewed;
  if (status === "needs_review") return Boolean(labels.needs_review);
  if (status === "cultural") return (item.reference_terms || []).length > 0;
  return true;
}

function applyFilters(keepSelection = true) {
  const selectedId = state.selected?.image_id;
  state.filtered = state.items.filter(itemMatchesFilter);
  if (keepSelection && selectedId !== undefined) {
    const idx = state.filtered.findIndex((item) => item.image_id === selectedId);
    state.currentIndex = idx >= 0 ? idx : 0;
  } else {
    state.currentIndex = 0;
  }
  renderCurrent();
}

function renderList() {
  els.itemList.innerHTML = "";
  state.filtered.forEach((item, index) => {
    const labels = labelsFor(item);
    const row = document.createElement("button");
    row.type = "button";
    row.className = `item-row${index === state.currentIndex ? " active" : ""}`;
    row.addEventListener("click", () => {
      state.currentIndex = index;
      renderCurrent();
    });

    const id = document.createElement("div");
    id.className = "item-id";
    id.textContent = `#${item.image_id}`;

    const summary = document.createElement("div");
    summary.className = "item-summary";
    const terms = termNames(item.reference_terms);
    summary.textContent = terms.length ? terms.join(", ") : item.prediction;

    const status = document.createElement("div");
    status.className = "item-status";
    if (labels.needs_review) status.classList.add("needs-review");
    else if (labels.reviewed) status.classList.add("reviewed");

    row.appendChild(id);
    row.appendChild(summary);
    row.appendChild(status);
    els.itemList.appendChild(row);
  });
}

function collectAnnotation() {
  const item = state.selected;
  return {
    image_id: item.image_id,
    image_filename: item.image_filename,
    prediction: item.prediction,
    references: item.references,
    reference_terms: item.reference_terms,
    prediction_terms: item.prediction_terms,
    ai_draft: item.ai_draft,
    annotator: els.annotatorInput.value.trim(),
    labels: {
      reviewed: els.reviewedInput.checked,
      needs_review: els.needsReviewInput.checked,
      caption_quality: state.segmentValues.caption_quality,
      specificity: state.segmentValues.specificity,
      cultural_entity_missed: els.culturalMissedInput.checked,
      template_bias: els.templateBiasInput.checked,
      object_hallucination: els.hallucinationInput.checked,
      wrong_object_or_action: els.wrongObjectInput.checked,
      language_issue: els.languageIssueInput.checked,
    },
    expected_cultural_terms: inputToTerms(els.expectedTermsInput.value),
    predicted_cultural_terms: inputToTerms(els.predictedTermsInput.value),
    explanation: els.explanationInput.value.trim(),
  };
}

async function saveCurrent() {
  if (!state.selected) return;
  els.saveState.textContent = "Saving";
  const payload = collectAnnotation();
  const res = await fetch("/api/annotation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!data.ok) {
    els.saveState.textContent = data.error || "Save failed";
    return;
  }

  const idx = state.items.findIndex((item) => item.image_id === payload.image_id);
  if (idx >= 0) {
    state.items[idx].annotation = {
      ...payload,
      updated_at: data.saved_at,
    };
  }
  state.progress = data.progress;
  els.saveState.textContent = `Saved ${data.saved_at}`;
  updateProgress();
  applyFilters(true);
}

function move(delta) {
  if (!state.filtered.length) return;
  state.currentIndex = Math.min(Math.max(state.currentIndex + delta, 0), state.filtered.length - 1);
  renderCurrent();
}

async function loadData() {
  els.progressText.textContent = "Loading";
  const res = await fetch("/api/items");
  const data = await res.json();
  state.items = data.items || [];
  state.outputs = data.outputs || {};
  state.progress = data.progress || state.progress;
  updateProgress();
  applyFilters(false);
}

function bindEvents() {
  els.refreshBtn.addEventListener("click", loadData);
  els.searchInput.addEventListener("input", () => applyFilters(true));
  els.statusFilter.addEventListener("change", () => applyFilters(true));
  els.prevBtn.addEventListener("click", () => move(-1));
  els.nextBtn.addEventListener("click", () => move(1));
  els.applyDraftBtn.addEventListener("click", applyDraftToForm);
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    saveCurrent();
  });
  els.annotatorInput.addEventListener("input", () => {
    localStorage.setItem("vig_annotation_annotator", els.annotatorInput.value);
  });

  document.querySelectorAll(".segmented button").forEach((button) => {
    button.addEventListener("click", () => {
      const field = button.closest(".segmented").dataset.field;
      const next = state.segmentValues[field] === button.dataset.value ? "" : button.dataset.value;
      setSegment(field, next);
    });
  });

  window.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
  });
}

els.annotatorInput.value = localStorage.getItem("vig_annotation_annotator") || "";
bindEvents();
loadData();
