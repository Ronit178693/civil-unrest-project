const pptxgen = require("pptxgenjs");
const path = require("path");

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
  slide.addText("Civil Unrest Event Extraction & Prediction — Milestone 1", {
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
    x: 0.8, y: 1.9, w: 11.7, h: 2.1, fontSize: 36, bold: true, color: "FFFFFF",
    fontFace: "Calibri", align: "left", isTextBox: true, margin: 0,
  });
  s.addText("Milestone 1 — Problem Domain, Feasibility, Objectives & Methodology, Algorithmic Relevance", {
    x: 0.8, y: 4.0, w: 11, h: 0.5, fontSize: 16, color: "CFE0F0", italics: true, isTextBox: true, margin: 0,
  });
  s.addText("Project Instructor: Dr. Pratima Singh  •  Dept. of Computer Science Engineering, Bennett University", {
    x: 0.8, y: 6.6, w: 11, h: 0.4, fontSize: 13, color: "9FB3C8", isTextBox: true, margin: 0,
  });
}

// ---------- Slide 2: Problem Domain Identification and Analysis ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "1. Problem Domain: Identification & Analysis");
  bulletBlock(s, [
    { text: "Domain", bold: true, color: NAVY },
    "Civil unrest — protests, riots, strikes, demonstrations — is organized and discussed on social media before and during the event",
    "Twitter/X and Reddit carry early-warning signals: grievance narratives, mobilization calls, on-ground reports",
    { text: "Problem", bold: true, color: NAVY },
    "Event-extraction models are trained on formal corpora (news, government data) and degrade on informal, noisy, emotionally charged, code-mixed social text",
    "Most existing systems stop at extraction — they do not model how a sequence of extracted events can forecast escalation",
  ], { x: 0.6, y: 1.35, w: 11.8, h: 2.9, fontSize: 16 });

  s.addShape("roundRect", { x: 0.6, y: 4.5, w: 11.8, h: 2.0, fill: { color: LIGHT }, line: { color: NAVY, width: 1 }, rectRadius: 0.08 });
  s.addText([
    { text: "Research Question\n", options: { bold: true, fontSize: 16, color: NAVY, breakLine: true } },
    { text: "How can event extraction from informal social media text be adapted to reliably identify civil-unrest-specific triggers and arguments, and how effectively can the resulting event sequences predict the emergence or escalation of civil unrest?", options: { fontSize: 15, italics: true, color: "222222" } },
  ], { x: 0.9, y: 4.65, w: 11.2, h: 1.7, isTextBox: true, margin: 0, valign: "middle" });
}

// ---------- Slide 3: Review of Existing Systems & Feasibility ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "2. Existing Systems & Project Feasibility");
  bulletBlock(s, [
    { text: "Existing systems / related work", bold: true, color: NAVY },
    "Classical event-extraction pipelines (news/formal text) and the project brief's proposed fine-tuned transformer taggers (BERTweet, XLM-RoBERTa) for noisy, code-mixed social text",
    "GDELT (EventRootCode 14 / PROTEST) and ACLED are standard weak-supervision / ground-truth sources for real-world unrest events, used as the benchmark targets",
    { text: "Feasibility", bold: true, color: NAVY },
    "No GPU / HuggingFace model-hub access in the offline environment → the transformer tagger is substituted with a linear-chain CRF (sklearn-crfsuite) that trains in seconds on CPU",
    "A documented TransformerTaggerStub preserves the exact upgrade path, so the architecture remains feasible to scale once GPU access is available",
    "A structurally realistic synthetic dataset (2,192 posts, 16 regions) lets the full pipeline be built, tested, and evaluated end-to-end today, with a direct swap-in path to real Twitter/Reddit + GDELT/ACLED data",
  ], { x: 0.6, y: 1.35, w: 11.8, h: 5.4, fontSize: 15.5 });
}

// ---------- Slide 4: Objectives and Methodology ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "3. Objectives & Methodology", "Four-stage pipeline");
  bulletBlock(s, [
    "Extract structured event frames (trigger, actor, location, target) from raw social posts",
    "Aggregate extracted events into per-region, per-time-window sequences",
    "Predict the probability of escalation in each region's next time window",
    "Evaluate with standard metrics plus lead-time analysis — how far in advance unrest can be anticipated",
  ], { x: 0.6, y: 1.35, w: 11.8, h: 1.9, fontSize: 15 });

  const boxes = [
    { title: "1. Extraction", desc: "Lexicon-informed CRF tags TRIGGER / ACTOR / LOCATION / TARGET per post", color: NAVY },
    { title: "2. Aggregation", desc: "Events bucketed per region x time window; stage counts + momentum features", color: GREEN },
    { title: "3. Prediction", desc: "Gradient-boosted trees forecast escalation in the next window, per region", color: RED },
    { title: "4. Evaluation", desc: "P/R/F1, ROC-AUC, and lead-time analysis (days of early warning)", color: GOLD },
  ];
  const startX = 0.6, boxW = 2.85, gap = 0.25, y = 3.5, h = 2.6;
  boxes.forEach((b, i) => {
    const x = startX + i * (boxW + gap);
    s.addShape("roundRect", { x, y, w: boxW, h, fill: { color: b.color }, rectRadius: 0.06 });
    s.addText(b.title, { x: x + 0.15, y: y + 0.2, w: boxW - 0.3, h: 0.5, fontSize: 16, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
    s.addText(b.desc, { x: x + 0.15, y: y + 0.8, w: boxW - 0.3, h: h - 1.0, fontSize: 12.5, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top" });
    if (i < boxes.length - 1) {
      s.addText("→", { x: x + boxW, y: y + h / 2 - 0.3, w: gap, h: 0.6, fontSize: 24, bold: true, color: "333333", align: "center", isTextBox: true, margin: 0 });
    }
  });
}

// ---------- Slide 5: Relevance of Algorithms / Techniques ----------
{
  const s = pres.addSlide();
  addTitleBar(s, "4. Relevance of Algorithms & Techniques");
  bulletBlock(s, [
    { text: "Extraction — Linear-chain CRF (BIO tagging)", bold: true, color: NAVY },
    "Models label dependencies across a token sequence, a natural fit for span-style roles (TRIGGER/ACTOR/LOCATION/TARGET); far cheaper than a transformer while offline, with features = lexicon hits, word shape, local context",
    { text: "Prediction — Gradient-Boosted Trees", bold: true, color: NAVY },
    "Handles the hand-crafted temporal/trend features (stage counts, deltas, rolling means, actor/target diversity) well on a small, tabular, region-grouped dataset; region-grouped splitting prevents leakage across regions",
    { text: "Ablation — From-scratch NumPy GRU", bold: true, color: NAVY },
    "A lightweight sequence encoder included as a teaching/ablation reference for learned (vs. hand-crafted) temporal representations",
    { text: "Why these are relevant", bold: true, color: RED },
    "Each technique directly targets a property of the data — sequence labeling for noisy text spans, tree ensembles for small structured feature sets, and a documented path to transformer/RNN upgrades as data and compute scale",
  ], { x: 0.6, y: 1.35, w: 11.8, h: 5.5, fontSize: 14.5 });
}

pres.writeFile({ fileName: path.join(__dirname, "civil_unrest_milestone1_slides.pptx") }).then(() => {
  console.log("Wrote civil_unrest_milestone1_slides.pptx");
});
