const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = (name) => path.join(__dirname, "..", "paper", "figures", name);

const NAVY = "1F3A5F";
const RED = "C94C4C";
const GREEN = "3F8F6B";
const GOLD = "E0A62E";
const GREY = "555555";
const LIGHT = "F4F6F8";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 in

function addTitleBar(slide, title, kicker) {
  slide.addShape("rect", { x: 0, y: 0, w: 13.33, h: 1.05, fill: { color: NAVY } });
  slide.addText(title, {
    x: 0.5, y: 0.12, w: 12.3, h: 0.6, fontSize: 26, bold: true, color: "FFFFFF",
    fontFace: "Calibri", isTextBox: true, margin: 0,
  });
  if (kicker) {
    slide.addText(kicker, {
      x: 0.5, y: 0.62, w: 12.3, h: 0.35, fontSize: 13, color: "CFE0F0",
      fontFace: "Calibri", isTextBox: true, margin: 0,
    });
  }
  slide.addText("Civil Unrest Event Extraction & Prediction", {
    x: 0.5, y: 7.15, w: 8, h: 0.3, fontSize: 9, color: "999999", isTextBox: true, margin: 0,
  });
}

function bulletBlock(slide, items, opts) {
  const paras = items.map((t) => ({
    text: typeof t === "string" ? t : t.text,
    options: {
      bullet: { code: "25AA", indent: 18 },
      fontSize: (typeof t === "object" && t.size) || opts.fontSize || 16,
      color: (typeof t === "object" && t.color) || "222222",
      breakLine: true,
      paraSpaceAfter: 10,
      indentLevel: (typeof t === "object" && t.level) || 0,
      bold: (typeof t === "object" && t.bold) || false,
    },
  }));
  paras[paras.length - 1].options.breakLine = false;
  slide.addText(paras, { x: opts.x, y: opts.y, w: opts.w, h: opts.h, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
}

// ---------- Slide 1: Title ----------
{
  const s = pres.addSlide();
  s.addShape("rect", { x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: NAVY } });
  s.addShape("rect", { x: 0, y: 4.55, w: 13.33, h: 0.06, fill: { color: GOLD } });
  s.addText("Civil Unrest Event Extraction\nand Prediction from Social Media Data", {
    x: 0.8, y: 2.1, w: 11.7, h: 2.1, fontSize: 38, bold: true, color: "FFFFFF",
    fontFace: "Calibri", align: "left", isTextBox: true, margin: 0,
  });
  s.addText("Extracting civil-unrest events from informal social text and forecasting escalation from event sequences", {
    x: 0.8, y: 4.75, w: 10.5, h: 0.5, fontSize: 16, color: "CFE0F0", italics: true, isTextBox: true, margin: 0,
  });
  s.addText("Project Instructor: Dr. Pratima Singh  •  Dept. of Computer Science Engineering, Bennett University", {
    x: 0.8, y: 6.6, w: 11, h: 0.4, fontSize: 13, color: "9FB3C8", isTextBox: true, margin: 0,
  });
}

// ---------- Slide 2: Motivation ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Motivation", "Why social media, and why now");
  bulletBlock(s, [
    "Civil unrest — protests, riots, strikes, demonstrations — is organized and discussed on social media before and during it happens",
    "Twitter/X and Reddit surface early-warning signals: grievance narratives, mobilization calls, on-ground reports",
    "Early, structured insight into unrest has direct value for public safety planning, humanitarian response, and policy monitoring",
    { text: "The catch: this text is informal, emotionally charged, multilingual, and often deliberately vague to dodge moderation", bold: true, color: RED },
  ], { x: 0.6, y: 1.4, w: 7.1, h: 5.3, fontSize: 18 });

  s.addImage({ path: FIG("fig1_event_stage_timeline.png"), x: 7.9, y: 1.5, w: 5.0, h: 2.67 });
  s.addText("Real unrest timelines ramp through stages — exactly what we want to detect early", {
    x: 7.9, y: 4.25, w: 5.0, h: 0.6, fontSize: 12, italics: true, color: GREY, isTextBox: true, margin: 0,
  });
}

// ---------- Slide 3: Problem Statement ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Problem Statement");
  bulletBlock(s, [
    { text: "Existing event extraction is trained on formal corpora (news, government data)", bold: true },
    "It degrades sharply on informal, noisy, emotionally charged social media text",
    { text: "Most systems stop at extraction", bold: true },
    "They don't model how a sequence of extracted events can forecast escalation",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 2.3, fontSize: 19 });

  s.addShape("roundRect", { x: 0.6, y: 4.1, w: 11.8, h: 2.3, fill: { color: LIGHT }, line: { color: NAVY, width: 1 }, rectRadius: 0.08 });
  s.addText([
    { text: "Research Question\n", options: { bold: true, fontSize: 18, color: NAVY, breakLine: true } },
    { text: "How can event extraction from informal social media text be adapted to reliably identify civil-unrest-specific triggers and arguments, and how effectively can the resulting event sequences predict the emergence or escalation of civil unrest?", options: { fontSize: 17, italics: true, color: "222222" } },
  ], { x: 0.9, y: 4.3, w: 11.2, h: 1.9, isTextBox: true, margin: 0, valign: "middle" });
}

// ---------- Slide 4: System Architecture ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "System Architecture", "Three modular, swappable stages");

  const boxes = [
    { title: "1. Extraction", desc: "Lexicon-informed CRF tags TRIGGER / ACTOR / LOCATION / TARGET per post", color: NAVY },
    { title: "2. Aggregation", desc: "Events bucketed per region x time window; stage counts + momentum features", color: GREEN },
    { title: "3. Prediction", desc: "Gradient-boosted trees forecast escalation in the next window, per region", color: RED },
    { title: "4. Evaluation", desc: "P/R/F1, ROC-AUC, and lead-time analysis (days of early warning)", color: GOLD },
  ];
  const startX = 0.6, boxW = 2.85, gap = 0.25, y = 2.0, h = 2.6;
  boxes.forEach((b, i) => {
    const x = startX + i * (boxW + gap);
    s.addShape("roundRect", { x, y, w: boxW, h, fill: { color: b.color }, rectRadius: 0.06 });
    s.addText(b.title, { x: x + 0.15, y: y + 0.2, w: boxW - 0.3, h: 0.5, fontSize: 17, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
    s.addText(b.desc, { x: x + 0.15, y: y + 0.8, w: boxW - 0.3, h: h - 1.0, fontSize: 13, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top" });
    if (i < boxes.length - 1) {
      s.addText("\u2192", { x: x + boxW, y: y + h / 2 - 0.3, w: gap, h: 0.6, fontSize: 24, bold: true, color: "333333", align: "center", isTextBox: true, margin: 0 });
    }
  });

  s.addText("Posts in (synthetic demo / real Twitter+Reddit) \u2192 structured events \u2192 region/time aggregates \u2192 escalation forecast + lead time", {
    x: 0.6, y: 5.0, w: 11.8, h: 0.5, fontSize: 14, italics: true, color: GREY, isTextBox: true, margin: 0, align: "center",
  });
}

// ---------- Slide 5: Extraction Methodology ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Stage 1 — Event Extraction", "Linear-chain CRF over lexicon + orthographic features");
  bulletBlock(s, [
    "BIO tagging over 4 roles: TRIGGER, ACTOR, LOCATION, TARGET",
    "Civil-unrest trigger lexicon: protest, strike, clash, riot, sit-in, repression (tear gas, curfew...)",
    "Features: unigram + bigram lexicon hits, word shape/casing, hashtags, local context window",
    { text: "Why CRF, not a transformer (as the brief proposes)?", bold: true, color: NAVY },
    "No network path to pretrained weights in this environment \u2192 CRF trains offline in seconds, CPU-only",
    { text: "Upgrade path documented in code: BERTweet / XLM-R fine-tuning, same (tokens \u2192 BIO) interface", color: GREY },
  ], { x: 0.6, y: 1.5, w: 11.8, h: 5.2, fontSize: 17 });
}

// ---------- Slide 6: Prediction Methodology ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Stage 2 — Escalation Prediction", "From event sequences to a forecast");
  bulletBlock(s, [
    "Extracted events aggregated per region into rolling 3-day windows",
    "Features: stage counts (grievance/mobilization/on-ground/escalation), actor/target diversity, trend/momentum (deltas, rolling means)",
    "Gradient-boosted tree classifier predicts: will this region escalate in the NEXT window?",
    { text: "Region-grouped train/test split", bold: true, color: NAVY },
    "No region's baseline chatter leaks between train and test — model must generalize across regions",
    "NumPy GRU sequence model also included as a from-scratch ablation reference",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 5.2, fontSize: 17 });
}

// ---------- Slide 7: Dataset ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Dataset", "Structurally realistic synthetic benchmark");
  bulletBlock(s, [
    "2,192 posts across 16 regions, 21-day window",
    "6 regions constructed to escalate (grievance \u2192 mobilization \u2192 on-ground \u2192 escalation)",
    "10 regions held flat/noisy as negatives",
    "Gold BIO labels + escalation ground-truth dates generated alongside the text",
    { text: "Real-data path documented:", bold: true, color: NAVY },
    "Twitter/X API v2 + Reddit (PRAW) for text; GDELT GKG 2.0 / ACLED for ground-truth escalation labels",
  ], { x: 0.6, y: 1.5, w: 6.9, h: 5.3, fontSize: 17 });

  s.addImage({ path: FIG("fig1_event_stage_timeline.png"), x: 7.7, y: 1.6, w: 5.1, h: 2.72 });
  s.addText("Synthetic escalation ramp for a representative region", {
    x: 7.7, y: 4.35, w: 5.1, h: 0.4, fontSize: 12, italics: true, color: GREY, isTextBox: true, margin: 0,
  });
}

// ---------- Slide 8: Extraction Results ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Results — Extraction", "Token-level precision / recall / F1 by role");

  s.addImage({ path: FIG("fig2_extraction_scores.png"), x: 0.6, y: 1.5, w: 7.3, h: 3.9 });

  const rows = [
    [{ text: "Role", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
     { text: "P", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
     { text: "R", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
     { text: "F1", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } }],
    ["ACTOR", "1.000", "0.993", "0.997"],
    ["TRIGGER", "1.000", "0.994", "0.997"],
    ["TARGET", "1.000", "1.000", "1.000"],
    ["LOCATION", "1.000", "1.000", "1.000"],
    [{ text: "Weighted avg", options: { bold: true } }, "0.999", "0.998", "0.998"],
  ];
  s.addTable(rows, { x: 8.2, y: 1.6, w: 4.5, h: 2.4, fontSize: 13, border: { type: "solid", color: "CCCCCC", pt: 0.5 }, align: "center", valign: "middle" });
  s.addText("Scores are an upper bound on this template-generated corpus — Section VII of the paper discusses expected real-data performance.", {
    x: 8.2, y: 4.2, w: 4.5, h: 1.2, fontSize: 12, italics: true, color: GREY, isTextBox: true, margin: 0,
  });
}

// ---------- Slide 9: Prediction Results ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Results — Escalation Prediction", "Held-out, region-grouped test split");

  s.addImage({ path: FIG("fig4_roc_curve.png"), x: 0.7, y: 1.5, w: 4.3, h: 4.3 });
  s.addImage({ path: FIG("fig3_feature_importance.png"), x: 5.3, y: 1.7, w: 7.3, h: 3.9 });

  const metrics = ["Accuracy: 0.943", "Precision: 0.833", "Recall: 0.833", "F1: 0.833", "ROC-AUC: 0.980"];
  s.addText(metrics.join("    |    "), {
    x: 0.6, y: 5.9, w: 12.1, h: 0.5, fontSize: 15, bold: true, color: NAVY, align: "center", isTextBox: true, margin: 0,
  });
}

// ---------- Slide 10: Lead-Time Analysis ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Lead-Time Analysis", "How early does the model raise the alarm?");

  s.addImage({ path: FIG("fig5_lead_time.png"), x: 0.6, y: 1.5, w: 6.8, h: 4.2 });

  bulletBlock(s, [
    "For every region that truly escalates, find the first window the model's probability crosses the decision threshold",
    "Lead time = (true escalation window) \u2212 (first flagged window)",
    "2 of 3 held-out escalating regions detected at or before the true window",
    { text: "1 region missed at default threshold \u2192 headroom for threshold tuning / more features", color: RED },
  ], { x: 7.7, y: 1.6, w: 5.0, h: 4.3, fontSize: 15 });
}

// ---------- Slide 11: Limitations ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Limitations & Future Work");
  bulletBlock(s, [
    { text: "Limitations", bold: true, color: NAVY },
    "Synthetic benchmark inflates extraction scores relative to real, noisy text",
    "CRF substituted for a fine-tuned transformer due to offline compute constraints (upgrade path documented)",
    "Real-world escalation ground truth (ACLED/GDELT) lags events — real lead-time numbers will differ",
    { text: "Future Work", bold: true, color: NAVY },
    "Validate extraction on a small hand-labeled real Twitter/Reddit sample",
    "Fine-tune BERTweet / XLM-R once GPU + model-hub access is available",
    "Add sentiment intensity and cross-region diffusion features",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 5.3, fontSize: 17 });
}

// ---------- Slide 12: Conclusion / Deliverables ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "Conclusion & Deliverables");
  bulletBlock(s, [
    "End-to-end, fully working pipeline: extraction \u2192 aggregation \u2192 prediction \u2192 evaluation",
    "0.99 weighted extraction F1, 0.98 prediction ROC-AUC on the synthetic benchmark",
    "Early-warning capability demonstrated via explicit lead-time analysis",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 2.2, fontSize: 18 });

  const checklist = [
    ["\u2713", "Source Code", "GitHub-ready repository, tested with pytest"],
    ["\u2713", "Research Paper", "IEEE two-column format (.docx + .pdf)"],
    ["\u2713", "Presentation Slides", "This deck (.pptx)"],
  ];
  let y = 4.0;
  checklist.forEach((row) => {
    s.addShape("roundRect", { x: 0.6, y, w: 11.8, h: 0.75, fill: { color: LIGHT }, line: { color: "DDDDDD", width: 0.5 }, rectRadius: 0.05 });
    s.addText(row[0], { x: 0.8, y: y + 0.08, w: 0.5, h: 0.6, fontSize: 20, bold: true, color: GREEN, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(row[1], { x: 1.4, y: y + 0.08, w: 3.0, h: 0.6, fontSize: 15, bold: true, color: NAVY, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(row[2], { x: 4.5, y: y + 0.08, w: 7.7, h: 0.6, fontSize: 14, color: "333333", isTextBox: true, margin: 0, valign: "middle" });
    y += 0.95;
  });
}

// ---------- Slide 13: Thank you ----------
{
  const s = pres.addSlide();
  s.addShape("rect", { x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: NAVY } });
  s.addText("Thank You", { x: 0.8, y: 2.9, w: 11.7, h: 1.2, fontSize: 44, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
  s.addText("Questions & Discussion", { x: 0.8, y: 4.1, w: 11.7, h: 0.6, fontSize: 20, color: "CFE0F0", isTextBox: true, margin: 0 });
  s.addText("Project Instructor: Dr. Pratima Singh  •  Pratima.singh1@bennett.edu.in", {
    x: 0.8, y: 6.7, w: 11, h: 0.4, fontSize: 13, color: "9FB3C8", isTextBox: true, margin: 0,
  });
}

pres.writeFile({ fileName: path.join(__dirname, "civil_unrest_presentation.pptx") }).then(() => {
  console.log("Wrote civil_unrest_presentation.pptx");
});
