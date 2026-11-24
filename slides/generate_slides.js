// Generates slides/civil_unrest_presentation.pptx.
// Run:  node slides/generate_slides.js      (needs `npm install pptxgenjs`)
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
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in

function addTitleBar(slide, title, kicker) {
  slide.addShape("rect", { x: 0, y: 0, w: 13.33, h: 1.05, fill: { color: NAVY } });
  slide.addText(title, { x: 0.5, y: 0.12, w: 12.3, h: 0.6, fontSize: 26, bold: true, color: "FFFFFF", fontFace: "Calibri", isTextBox: true, margin: 0 });
  if (kicker) {
    slide.addText(kicker, { x: 0.5, y: 0.62, w: 12.3, h: 0.35, fontSize: 13, color: "CFE0F0", fontFace: "Calibri", isTextBox: true, margin: 0 });
  }
  slide.addText("Civil Unrest Event Extraction & Prediction", { x: 0.5, y: 7.15, w: 8, h: 0.3, fontSize: 9, color: "999999", isTextBox: true, margin: 0 });
}

function bulletBlock(slide, items, opts) {
  const paras = items.map((t) => ({
    text: typeof t === "string" ? t : t.text,
    options: {
      bullet: { code: "25AA", indent: 18 },
      fontSize: (typeof t === "object" && t.size) || opts.fontSize || 16,
      color: (typeof t === "object" && t.color) || "222222",
      breakLine: true, paraSpaceAfter: 9,
      indentLevel: (typeof t === "object" && t.level) || 0,
      bold: (typeof t === "object" && t.bold) || false,
    },
  }));
  paras[paras.length - 1].options.breakLine = false;
  slide.addText(paras, { x: opts.x, y: opts.y, w: opts.w, h: opts.h, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
}

const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } });
const bold = (t) => ({ text: t, options: { bold: true } });
function table(slide, rows, opts) {
  slide.addTable(rows, { fontSize: 12, border: { type: "solid", color: "CCCCCC", pt: 0.5 }, align: "center", valign: "middle", fontFace: "Calibri", ...opts });
}
function caption(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.4, fontSize: 11, italics: true, color: GREY, isTextBox: true, margin: 0 });
}

// 1. Title
{
  const s = pres.addSlide();
  s.addShape("rect", { x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: NAVY } });
  s.addShape("rect", { x: 0, y: 4.55, w: 13.33, h: 0.06, fill: { color: GOLD } });
  s.addText("Civil Unrest Event Extraction\nand Prediction from Social Media Data", { x: 0.8, y: 2.1, w: 11.7, h: 2.1, fontSize: 38, bold: true, color: "FFFFFF", fontFace: "Calibri", isTextBox: true, margin: 0 });
  s.addText("A fine-tuned transformer extracts unrest events from informal text; event sequences forecast escalation, tested on real GDELT and Mastodon data", { x: 0.8, y: 4.75, w: 11.5, h: 0.7, fontSize: 16, color: "CFE0F0", italics: true, isTextBox: true, margin: 0 });
  s.addText("Ronit Agrawal  •  Project Instructor: Dr. Pratima Singh  •  Dept. of Computer Science Engineering, Bennett University", { x: 0.8, y: 6.6, w: 11.8, h: 0.4, fontSize: 13, color: "9FB3C8", isTextBox: true, margin: 0 });
}

// 2. Motivation
{
  const s = pres.addSlide();
  addTitleBar(s, "Motivation", "Why social media, and why it is hard");
  bulletBlock(s, [
    "Protests, strikes and riots are organised and discussed on social media before and while they happen",
    "Posts carry early signals: grievance narratives, mobilisation calls, on-ground reports",
    "Early, structured warning helps public-safety planning, humanitarian response and policy monitoring",
    { text: "The catch: this text is informal, emotional, often multilingual and deliberately vague, unlike the news text most extractors are trained on", bold: true, color: RED },
  ], { x: 0.6, y: 1.4, w: 6.9, h: 5.3, fontSize: 18 });
  s.addImage({ path: FIG("fig1_event_stage_timeline.png"), x: 7.7, y: 1.6, w: 5.3, h: 2.12 });
  caption(s, "Unrest ramps through stages: grievance → mobilisation → on-ground → escalation (synthetic example)", 7.7, 3.8, 5.3);
}

// 3. Problem statement
{
  const s = pres.addSlide();
  addTitleBar(s, "Problem Statement");
  bulletBlock(s, [
    { text: "Event extractors trained on formal corpora degrade on informal social media text", bold: true },
    "(a) Extract unrest triggers and arguments: actor, location, time, event type, scale, target",
    { text: "Most systems stop at extraction", bold: true },
    "(b) Use the extracted event sequences to forecast whether escalation starts in a region within 3 days",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 2.4, fontSize: 19 });
  s.addShape("roundRect", { x: 0.6, y: 4.1, w: 11.8, h: 2.3, fill: { color: LIGHT }, line: { color: NAVY, width: 1 }, rectRadius: 0.08 });
  s.addText([
    { text: "Research Question\n", options: { bold: true, fontSize: 18, color: NAVY, breakLine: true } },
    { text: "How can event extraction from informal social media text be adapted to reliably identify civil-unrest-specific triggers and arguments, and how effectively can the resulting event sequences predict the emergence or escalation of civil unrest?", options: { fontSize: 17, italics: true, color: "222222" } },
  ], { x: 0.9, y: 4.3, w: 11.2, h: 1.9, isTextBox: true, margin: 0, valign: "middle" });
}

// 4. Architecture
{
  const s = pres.addSlide();
  addTitleBar(s, "System Architecture", "Two stages, connected only through extracted events");
  const boxes = [
    { title: "1. Extraction", desc: "Fine-tuned Twitter-RoBERTa tags trigger, actor, location, target, time, scale; event type from the trigger. Stage classifier: grievance → escalation", color: NAVY },
    { title: "2. Aggregation", desc: "Per region and day: 3-day windows, 30 features (stage counts, actors, targets, momentum). Windows during escalation excluded", color: GREEN },
    { title: "3. Forecasting", desc: "Volume baseline, logistic regression, gradient boosting, GRU sequence model: P(escalation starts in next 3 days)", color: RED },
    { title: "4. Evaluation", desc: "Span F1; precision, recall, ROC-AUC, average precision; onsets warned, lead time, false alarms", color: GOLD },
  ];
  const startX = 0.6, boxW = 2.85, gap = 0.25, y = 1.7, h = 3.0;
  boxes.forEach((b, i) => {
    const x = startX + i * (boxW + gap);
    s.addShape("roundRect", { x, y, w: boxW, h, fill: { color: b.color }, rectRadius: 0.06 });
    s.addText(b.title, { x: x + 0.15, y: y + 0.2, w: boxW - 0.3, h: 0.5, fontSize: 17, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
    s.addText(b.desc, { x: x + 0.15, y: y + 0.8, w: boxW - 0.3, h: h - 1.0, fontSize: 13, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top" });
    if (i < boxes.length - 1) {
      s.addText("→", { x: x + boxW, y: y + h / 2 - 0.3, w: gap, h: 0.6, fontSize: 24, bold: true, color: "333333", align: "center", isTextBox: true, margin: 0 });
    }
  });
  s.addText("The forecaster never sees raw text or GDELT codes: only what Stage 1 extracted, so every alarm traces back to who, where and over what", { x: 0.6, y: 5.1, w: 11.8, h: 0.6, fontSize: 14, italics: true, color: GREY, isTextBox: true, margin: 0, align: "center" });
}

// 5. Data
{
  const s = pres.addSlide();
  addTitleBar(s, "Data", "Two real sources, synthetic data for controlled tests");
  table(s, [
    [hdr("Source"), hdr("Size"), hdr("Used for")],
    [bold("GDELT events, India, May–Oct 2026"), "28,362 protest events; 11,905 headlines; 22 states; 62 escalation onsets", "Real forecasting: headlines from article URLs as text, violent-protest codes (CAMEO 145) as truth"],
    [bold("Mastodon public posts"), "1,188 English posts, 12 unrest hashtags; opted-in accounts only", "Real informal text for extraction"],
    [bold("Hand-annotated real set"), "160 items (80 posts + 80 headlines), 6 roles + stage", "80 to adapt, 80 to test Stage 1"],
    [bold("Synthetic posts"), "14,374 posts, 80 regions × 21 days, full gold labels", "Training; controlled forecasting"],
    [bold("Challenge set"), "24 hand-written posts, unseen vocabulary", "Generalisation test"],
  ], { x: 0.6, y: 1.4, w: 12.1, colW: [3.2, 4.4, 4.5], fontSize: 13, align: "left" });
  s.addText("Twitter/X, Facebook and Reddit need paid or approved API access (Reddit answers HTTP 403), so Mastodon stands in for social media.", { x: 0.6, y: 4.5, w: 12.1, h: 0.5, fontSize: 13, italics: true, color: RED, isTextBox: true, margin: 0 });
}

// 6. Extraction method
{
  const s = pres.addSlide();
  addTitleBar(s, "Stage 1: Event Extraction", "Fine-tuned transformer, with a CRF baseline");
  bulletBlock(s, [
    { text: "Model: cardiffnlp/twitter-roberta-base (RoBERTa further pre-trained on ~58M tweets)", bold: true, color: NAVY },
    "Token classification, BIO tags: TRIGGER, ACTOR, LOCATION, TARGET, TIME, SCALE",
    "Event type (protest, strike, clash, riot, sit-in, repression) read off the trigger via a domain lexicon",
    "3 epochs, AdamW, learning rate 3×10⁻⁵, CPU only (~1 h for the full run)",
    { text: "Baseline: linear-chain CRF over lexicon, gazetteer, time words, word shape, context", bold: true, color: NAVY },
    { text: "Domain adaptation: each model trained on synthetic only vs synthetic + 80 real items (×5)", bold: true, color: NAVY },
    "Stage classifier (TF-IDF + logistic regression): grievance / mobilisation / on-ground / escalation",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 5.3, fontSize: 17 });
}

// 7. Forecasting method
{
  const s = pres.addSlide();
  addTitleBar(s, "Stage 2: Escalation Forecasting", "From event sequences to an early warning");
  bulletBlock(s, [
    "Window = days e−3 to e−1 of a region; at the start of day e, alarm or not",
    "Label = a new escalation starts on days e to e+2; windows already in an escalation are excluded (no leakage)",
    "Real truth: ≥2 distinct articles report violent protest (CAMEO 145) in a state after 7 calm days",
    { text: "Models: volume baseline, logistic regression, gradient boosting, GRU (NumPy, backprop through time)", bold: true, color: NAVY },
    "Synthetic: 5-fold cross-validation grouped by region",
    "Real: time split, train May–Aug, test Sep–Oct 2026 (1,202 windows, 19 onsets)",
    "Reference: the same models trained on GDELT's own event codes instead of our extracted features",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 5.3, fontSize: 17 });
}

// 8. Extraction results
{
  const s = pres.addSlide();
  addTitleBar(s, "Results: Extraction", "80 real training items matter more than the model");
  s.addImage({ path: FIG("fig2_extraction_scores.png"), x: 0.5, y: 1.4, w: 7.6, h: 3.85 });
  table(s, [
    [hdr("Span F1, real test (80)"), hdr("Synthetic only"), hdr("Adapted")],
    ["CRF", "0.43", "0.58"],
    ["Transformer", "0.37", bold("0.63")],
  ], { x: 8.4, y: 1.5, w: 4.4, colW: [2.0, 1.2, 1.2], fontSize: 13 });
  bulletBlock(s, [
    "Synthetic-only models collapse on real text",
    "Adapted transformer is best overall; best on unseen vocabulary (0.86)",
    "Real-text F1 by role: trigger 0.78, actor 0.63, location 0.62, time 0.50, target 0.42, scale 0.40",
    "Stage accuracy on real text: 0.50 → 0.64",
  ], { x: 8.4, y: 3.0, w: 4.4, h: 3.6, fontSize: 14 });
}

// 9. Synthetic forecasting
{
  const s = pres.addSlide();
  addTitleBar(s, "Results: Forecasting on Synthetic Data", "64 regions, region-grouped 5-fold cross-validation");
  s.addImage({ path: FIG("fig4_pr_curves.png"), x: 0.5, y: 1.4, w: 5.2, h: 4.2 });
  table(s, [
    [hdr("Model"), hdr("F1"), hdr("AP"), hdr("Lead"), hdr("False alarm")],
    ["Volume baseline", "0.43", "0.49", "6.2 d", "40%"],
    ["Logistic regression", "0.87", "0.98", "1.9 d", "25%"],
    [bold("Gradient boosting"), bold("0.93"), bold("0.99"), "1.8 d", bold("5%")],
    ["GRU", "0.88", "0.94", "2.0 d", "20%"],
  ], { x: 6.1, y: 1.6, w: 6.7, colW: [2.3, 1.0, 1.0, 1.1, 1.3], fontSize: 13 });
  bulletBlock(s, [
    "Every escalating region warned before onset",
    "Event features beat volume by a wide margin; volume warns early only by alarming 40% of quiet regions",
    "Top feature: rolling count of posts with a trigger, then change in on-ground posts",
    { text: "~2-day lead is a ceiling set by the generator", color: GREY },
  ], { x: 6.1, y: 4.0, w: 6.7, h: 2.8, fontSize: 14 });
}

// 10. Real forecasting
{
  const s = pres.addSlide();
  addTitleBar(s, "Results: Forecasting on Real Data", "GDELT India, test period September–October 2026, base rate 4.4%");
  s.addImage({ path: FIG("fig6_real_pr_curves.png"), x: 0.5, y: 1.4, w: 5.4, h: 4.05 });
  table(s, [
    [hdr("Model (features)"), hdr("ROC-AUC"), hdr("AP"), hdr("Onsets warned")],
    ["Volume baseline", "0.62", "0.06", "11/19"],
    ["Logistic regr. (ours)", "0.63", "0.06", "13/19"],
    ["Gradient boosting (ours)", "0.60", "0.06", "0/19"],
    [bold("GRU (ours)"), "0.56", bold("0.13"), "5/19"],
    ["Logistic regr. (GDELT codes)", "0.70", "0.09", "14/19"],
  ], { x: 6.2, y: 1.5, w: 6.6, colW: [2.9, 1.1, 0.9, 1.7], fontSize: 13 });
  bulletBlock(s, [
    { text: "Real escalation is far harder: best AP 0.13, ~3× chance", bold: true, color: RED },
    "Logistic regression warns before 13 of 19 onsets, but only 8% of its alarms are right",
    "GDELT's own codes do only slightly better → the limit is predictability, not extraction",
  ], { x: 6.2, y: 4.3, w: 6.6, h: 2.5, fontSize: 14 });
}

// 11. Why real is hard
{
  const s = pres.addSlide();
  addTitleBar(s, "Why Real Escalation Is Hard to Forecast");
  s.addImage({ path: FIG("fig7_real_timeline.png"), x: 0.6, y: 1.3, w: 12.1, h: 4.84 });
  s.addText("Uttar Pradesh: escalations often arrive without a visible build-up in protest coverage; news reports events after they happen, while the social posts that might carry earlier signals are the data we could not access at scale.", { x: 0.6, y: 6.25, w: 12.1, h: 0.8, fontSize: 14, italics: true, color: GREY, isTextBox: true, margin: 0 });
}

// 12. Limitations and future work
{
  const s = pres.addSlide();
  addTitleBar(s, "Limitations & Future Work");
  bulletBlock(s, [
    { text: "Deviations from the brief (data access)", bold: true, color: NAVY },
    "Mastodon instead of Twitter/X, Facebook, Reddit; news headlines (not posts) for real forecasting; English only",
    { text: "Limitations", bold: true, color: NAVY },
    "GDELT auto-coding is noisy; 160 real items labelled by one annotator; alarm threshold fixed at 0.5",
    { text: "Future work", bold: true, color: NAVY },
    "ACLED + Reddit/X research access; 500+ annotated real items with a second annotator",
    "Multilingual (XLM-R) for Hindi and code-mixed posts; threshold calibration; tone, actor-type and neighbouring-state features",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 5.3, fontSize: 17 });
}

// 13. Conclusion and deliverables
{
  const s = pres.addSlide();
  addTitleBar(s, "Conclusion & Deliverables");
  bulletBlock(s, [
    "End-to-end pipeline: transformer extraction → aggregation → four forecasters → early-warning evaluation",
    "Domain adaptation is the key lever: real-text span F1 0.37 → 0.63 with 80 real items",
    "Near-perfect on synthetic data; weak but real signal on real GDELT data (AP 0.13 vs 0.04)",
  ], { x: 0.6, y: 1.5, w: 11.8, h: 2.2, fontSize: 18 });
  const checklist = [
    ["✓", "Source Code", "github.com/Ronit178693/civil-unrest-project, 18 pytest tests"],
    ["✓", "Research Paper", "IEEE two-column format (.docx + .pdf)"],
    ["✓", "Presentation Slides", "This deck (.pptx)"],
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

// 14. Thank you
{
  const s = pres.addSlide();
  s.addShape("rect", { x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: NAVY } });
  s.addText("Thank You", { x: 0.8, y: 2.9, w: 11.7, h: 1.2, fontSize: 44, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
  s.addText("Questions & Discussion", { x: 0.8, y: 4.1, w: 11.7, h: 0.6, fontSize: 20, color: "CFE0F0", isTextBox: true, margin: 0 });
  s.addText("Ronit Agrawal  •  Project Instructor: Dr. Pratima Singh", { x: 0.8, y: 6.7, w: 11, h: 0.4, fontSize: 13, color: "9FB3C8", isTextBox: true, margin: 0 });
}

pres.writeFile({ fileName: path.join(__dirname, "civil_unrest_presentation.pptx") }).then(() => {
  console.log("Wrote civil_unrest_presentation.pptx");
});
