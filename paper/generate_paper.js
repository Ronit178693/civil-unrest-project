// Generates paper/civil_unrest_paper.docx (IEEE-style, two columns).
// Run:  node paper/generate_paper.js      (needs `npm install docx`)
// The PDF is exported from the .docx (Word: File > Save as PDF).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, SectionType,
  convertInchesToTwip,
} = require("docx");

const FIG = (name) => path.join(__dirname, "figures", name);

// ---------- helpers ----------
const COL_WIDTH_PX = 330; // ~3.4in column at 96dpi

function img(name, pxW, pxH, caption) {
  const data = fs.readFileSync(FIG(name));
  const scale = COL_WIDTH_PX / pxW;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      children: [new ImageRun({ data, transformation: { width: Math.round(pxW * scale), height: Math.round(pxH * scale) }, type: "png" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, size: 16 })],
    }),
  ];
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 220, after: 100 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: text.toUpperCase(), bold: true, size: 20 })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 140, after: 60 },
    children: [new TextRun({ text, bold: true, italics: true, size: 20 })],
  });
}
function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 100, line: 252 },
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.noIndent ? undefined : { firstLine: 200 },
    children: [new TextRun({ text, size: 20 })],
  });
}
function bullet(text) {
  return new Paragraph({
    spacing: { after: 50 },
    bullet: { level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 20 })],
  });
}
function refItem(num, text) {
  return new Paragraph({
    spacing: { after: 60 },
    indent: { left: 300, hanging: 300 },
    children: [new TextRun({ text: `[${num}] `, size: 16 }), new TextRun({ text, size: 16 })],
  });
}
function tableCaption(text) {
  return new Paragraph({
    spacing: { before: 120, after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, size: 16, bold: true })],
  });
}

function metricsTable(headers, rows, firstColWide = true) {
  const n = headers.length;
  const w = (i) => (firstColWide ? (i === 0 ? 34 : Math.floor(66 / (n - 1))) : Math.floor(100 / n));
  const mkCell = (t, i, opts = {}) => new TableCell({
    width: { size: w(i), type: WidthType.PERCENTAGE },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F3A5F" } : undefined,
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(t), size: 15, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000" })],
    })],
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({ children: headers.map((hdr, i) => mkCell(hdr, i, { header: true })) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => mkCell(c, i)) })),
    ],
  });
}

// ---------- Title block (single column) ----------
const titleSection = {
  properties: {
    page: {
      size: { width: convertInchesToTwip(8.5), height: convertInchesToTwip(11) },
      margin: { top: convertInchesToTwip(0.75), bottom: convertInchesToTwip(1.0), left: convertInchesToTwip(0.75), right: convertInchesToTwip(0.75) },
    },
    type: SectionType.CONTINUOUS,
  },
  children: [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 160 },
      children: [new TextRun({ text: "Civil Unrest Event Extraction and Prediction from Social Media Data", bold: true, size: 32 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 20 },
      children: [new TextRun({ text: "Ronit Agrawal (S24CSEU1323), Simran Soni (S24CSEU1322), Jashanjot Kaur (S24CSEU1335),", size: 20 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 20 },
      children: [new TextRun({ text: "Anushka Mathur (S24CSEU1319), Shreya Desai (S24CSEU1306)", size: 20 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 10 },
      children: [new TextRun({ text: "Department of Computer Science Engineering, Bennett University", size: 20, italics: true })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: "Project Instructor: Dr. Pratima Singh  |  November 2026  |  Code: github.com/Ronit178693/civil-unrest-project", size: 18, italics: true })],
    }),
  ],
};

// ---------- Body (two columns) ----------
const B = [];

B.push(
  new Paragraph({
    spacing: { after: 100, line: 252 }, alignment: AlignmentType.JUSTIFIED,
    children: [
      new TextRun({ text: "Abstract—", bold: true, italics: true, size: 18 }),
      new TextRun({ size: 18, bold: true, text:
        "Social media often carries early signals of civil unrest, but event extractors trained on formal text degrade on informal posts, and most systems stop at extraction. We build a two-stage pipeline: a fine-tuned transformer (Twitter-pretrained RoBERTa) extracts unrest triggers and six arguments (actor, location, target, time, scale, event type) plus an escalation stage per post, and the extracted events, aggregated per region and day, feed forecasters that predict whether escalation starts within three days. We evaluate on synthetic data with full gold labels and on real data: 28,362 GDELT-coded protest events in India (May–October 2026), whose headlines are the input text and whose violent-protest codes are the ground truth, and 1,188 public Mastodon posts, with 160 real items annotated by hand. Extractors trained only on synthetic posts fall to 0.37–0.43 span F1 on real text; adding 80 real training items raises the transformer to 0.63, the best of four extractors. Forecasting is near perfect on synthetic data (F1 0.93) but weak on real data: the best average precision is 0.13 against a 0.04 base rate, about the same as when GDELT's own event codes are the input. Real escalation, not extraction quality, is the main obstacle to early warning." }),
    ],
  }),
  new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text: "Index Terms—", bold: true, italics: true, size: 18 }),
      new TextRun({ text: "civil unrest, event extraction, domain adaptation, transformers, social media, GDELT, early warning, escalation forecasting", italics: true, size: 18 })],
  }),
);

// I. Introduction
B.push(
  h2("I. Introduction"),
  body("Protests, strikes and riots are rarely sudden. Before large unrest, grievances are voiced, calls to mobilize circulate and on-the-ground reports accumulate on platforms such as Twitter/X and Reddit. This makes social media a timely, if noisy, sensor for public-safety planning, humanitarian response and policy monitoring."),
  body("Two problems stand in the way. First, social media text is short, informal, code-mixed and sometimes deliberately vague, unlike the news and government text most event extractors are trained on. Second, extraction alone does not warn anyone: the extracted events must be turned into a forecast of what happens next, and judged by how early it warns."),
  body("This paper addresses both problems together. Our contributions are: (1) a fine-tuned transformer extractor for unrest triggers and six argument roles, compared with a feature-based CRF and measured on synthetic, hand-written and real text; (2) evidence that a small amount of real annotated data (80 items) matters more for real-text extraction than the choice of model; (3) a leakage-free forecasting setup that feeds extracted events to four forecasters, including a GRU sequence model; and (4) an evaluation on real GDELT data from India with a time-based split, which shows how much harder real escalation is to forecast than synthetic escalation."),
);

// II. Related Work
B.push(
  h2("II. Related Work"),
  body("Event extraction moved from pattern-based systems on newswire (MUC, ACE) to neural sequence labelling and transformer-based trigger and argument extraction [3], [9]. These systems perform well on formal text but degrade on social media, where tokenization, entity boundaries and trigger vocabulary differ; models pre-trained on tweets, such as BERTweet [4] and Twitter-RoBERTa [10], narrow that gap."),
  body("Structured event databases such as GDELT [1], coded with the CAMEO scheme [11], and the human-coded ACLED [2] provide large-scale ground truth for protest and conflict. They are built from news, not social media, and we use GDELT as a label source and as a source of real headlines."),
  body("Unrest forecasting from open sources, notably EMBERS [6], aggregates indicators per region and time to predict protest occurrence. We follow the same aggregate-then-forecast structure but feed the forecaster explicit extracted events (who, where, over what), so a forecast can be traced to its evidence."),
);

// III. Problem Statement
B.push(
  h2("III. Problem Statement"),
  body("Given a stream of informal posts tagged with a region and a date, the task is (a) to extract civil-unrest event triggers and arguments (actor, location, time, event type, scale, target) and an escalation stage from each post, and (b) to use the resulting event sequences to forecast, at the start of each day, whether a new escalation will begin in a region within the next three days."),
  body("Research question: How can event extraction from informal social media text be adapted to reliably identify civil-unrest triggers and arguments, and how effectively can the resulting event sequences predict the emergence or escalation of civil unrest?"),
);

// IV. Data
B.push(
  h2("IV. Data"),
  h3("A. Synthetic Posts"),
  body("A generator simulates 5 independent episodes × 16 Indian cities = 80 regions over 21 days (14,374 posts). In each episode 6 regions escalate, 5 build up and then fizzle out (hard negatives) and 5 stay flat; random “viral” days stop raw volume from giving the answer away. Posts follow stage-specific templates (grievance, mobilization, on-ground, escalation) and carry complete gold BIO labels. One template per stage is held out from extractor training to test unseen sentence structures (Fig. 1)."),
  ...img("fig1_event_stage_timeline.png", 1350, 540, "Fig. 1. Posts per day by stage in a synthetic region that escalates and one that builds up, then fizzles."),
  h3("B. Real News Events (GDELT)"),
  body("We downloaded every GDELT 1.0 daily event file from 1 May to 31 October 2026 and kept the protest events (CAMEO root code 14) geolocated in India: 28,362 events, 15,692 distinct state-day articles and 11,905 unique headlines across 22 states with at least 60 articles. Each event cites a news article; we recover its headline from the URL slug (e.g. “du-teachers-stage-protest-demanding-regularisation-of-ad-hoc-faculty”), which works for 93% of URLs. The headlines are the only text the forecaster sees."),
  body("Ground truth comes from GDELT's coding, which the models never see as input. A state escalates on a day when at least two distinct articles report a violent protest or riot (CAMEO 145) after at least seven calm days. This yields 62 onsets in 18 states, led by Uttar Pradesh (8), West Bengal (7), Delhi (6) and Bihar (6)."),
  h3("C. Real Social Media Posts (Mastodon)"),
  body("Twitter/X, Facebook and Reddit no longer allow unauthenticated access (Reddit answers HTTP 403; the X API is paid), so we used the public Mastodon API. We collected 1,188 English posts from 12 unrest hashtags (e.g. #protest, #strike, #generalstrike), only from non-bot accounts that opted in to search indexing; no usernames or profile data are stored."),
  h3("D. Annotated Sets"),
  body("We annotated 160 real items by hand, 80 Mastodon posts and 80 GDELT headlines (random samples), with all six roles and a stage, using the same conventions as the synthetic labels (hashtags are never labelled). Half of each source trains the adapted models and half is held out. The sample deliberately keeps noise, such as music spam, countdown bots and military uses of “strike” or “blockade”, so false positives are tested. A separate hand-written challenge set of 24 posts uses actors, targets, cities and trigger words the generator never produces."),
);

// V. Methodology
B.push(
  h2("V. Methodology"),
  h3("A. Event Extraction"),
  body("Each post is tokenized by one shared tokenizer (hashtags and hyphenated words kept whole) and tagged with BIO labels for TRIGGER, ACTOR, LOCATION, TARGET, TIME and SCALE. The event type (protest, strike, clash, riot, sit-in, repression) is read off the first trigger through a domain lexicon. Our main model fine-tunes cardiffnlp/twitter-roberta-base [10], RoBERTa [9] further pre-trained on about 58M tweets, as a token classifier: each word's label goes on its first subword, other subwords are ignored, and predicted sequences are repaired so an I- tag never follows O. We train for 3 epochs with AdamW (learning rate 3×10⁻⁵, batch 16, linear warm-up) on a CPU."),
  body("The baseline is a linear-chain CRF [3] over a trigger lexicon (unigrams and bigrams), a gazetteer of Indian places, time words, word shape, casing, hashtags, affixes and a ±2-token context. Each extractor is trained twice: on synthetic posts only, and adapted, i.e. on synthetic posts plus the 80 real training items oversampled five times."),
  h3("B. Escalation Indicators"),
  body("A post-level classifier assigns one of four stages, grievance, mobilization, on-ground or escalation, using TF-IDF word (1–2) and character (3–5) n-grams plus four cue features (any trigger, escalation-type trigger, future-tense wording, live wording) and class-balanced logistic regression. It is trained and adapted like the extractors."),
  h3("C. Aggregation and Labels"),
  body("For each region and day e, a window covers days e−3 to e−1; at the start of day e the model raises an alarm or not. The label is 1 if an escalation onset falls on days e to e+2. Windows that already contain an escalation day are excluded, so the model only forecasts new escalations from a calm state and never sees the escalation it is predicting. Each window has nine counts (posts, the four stages, distinct actors and targets, trigger posts, escalation-trigger posts) plus their change from the previous window, a 2-window rolling mean and three stage ratios: 30 features."),
  h3("D. Forecasters"),
  body("We compare: a volume-only baseline (logistic regression on post counts); logistic regression and gradient-boosted trees [7] on all 30 features; and a single-layer GRU [12] (16 units) implemented in NumPy with full backpropagation through time, which reads each region's sequence of windows and learns its own temporal pattern. Its gradients are checked numerically in the test suite."),
  h3("E. Evaluation Protocol"),
  body("Extraction uses exact-match span precision, recall and F1 (start, end and role must all match). Synthetic forecasting uses 5-fold cross-validation grouped by region; Stage 1 is trained on episode 0 only and forecasting uses episodes 1–4, so the extractor never sees the forecasting regions. Real forecasting uses a time split: train on windows whose labels end before 1 September 2026, test on September–October (1,202 windows, 53 positive, 19 onsets). We report precision, recall, F1, ROC-AUC, average precision (AP, preferred for rare positives) and early-warning metrics: the share of onsets warned before they happen, the mean lead time in days and, for synthetic data, the share of quiet regions that ever get a false alarm. As a reference, the real forecasters are also trained on GDELT's own CAMEO counts per window instead of our extracted features."),
);

// VI. Results
B.push(
  h2("VI. Results"),
  h3("A. Event Extraction"),
  body("Table I and Fig. 2 compare the four extractors. All are near perfect on synthetic posts, but trained on synthetic data alone they collapse on real text (0.37–0.43 span F1). Adding 80 real training items closes most of the gap: the adapted CRF reaches 0.58 and the adapted transformer 0.63, the best overall. The transformer also generalises best to unseen vocabulary (0.86 on the challenge set against 0.78 for the CRF), while the CRF stays slightly ahead on Mastodon posts and the transformer is clearly ahead on headlines."),
  tableCaption("TABLE I. SPAN F1 OF THE FOUR EXTRACTORS"),
  metricsTable(
    ["Test set (n)", "CRF syn.", "CRF adapt.", "Transf. syn.", "Transf. adapt."],
    [
      ["Synthetic, held-out templates (1,000)", "0.99", "0.99", "0.93", "0.94"],
      ["Challenge set (24)", "0.77", "0.78", "0.85", "0.86"],
      ["Real Mastodon posts (40)", "0.38", "0.61", "0.32", "0.58"],
      ["Real GDELT headlines (40)", "0.49", "0.55", "0.43", "0.67"],
      ["All real test items (80)", "0.43", "0.58", "0.37", "0.63"],
    ]
  ),
  ...img("fig2_extraction_scores.png", 1125, 570, "Fig. 2. Span F1 by test set: synthetic-only vs adapted CRF and transformer."),
  body("On the real test items the adapted transformer finds triggers well (F1 0.78), actors and locations moderately (0.63 and 0.62), and struggles with time (0.50), targets (0.42) and scale (0.40), which are often long or implicit in real text. Adaptation also raises stage-classification accuracy on real text from 0.50 to 0.64 (synthetic held-out templates: 0.85 to 0.98)."),
  h3("B. Forecasting on Synthetic Data"),
  body("Table II reports the forecasters on the extracted features of 64 synthetic regions (1,020 windows, 55 positive). Event features beat the volume baseline by a wide margin. Gradient boosting is best, with F1 0.93, AP 0.99 and false alarms in only 5% of quiet regions; every model warns of every escalating region before onset, about two days ahead. The volume baseline warns earliest only because it alarms 40% of quiet regions. Gold features give almost the same scores, so extraction errors cost little on synthetic text. The roughly two-day lead time is a ceiling set by the generator, in which on-ground activity starts three days before onset."),
  tableCaption("TABLE II. SYNTHETIC FORECASTING (REGION-GROUPED 5-FOLD CV)"),
  metricsTable(
    ["Model", "Prec.", "Recall", "F1", "AP", "Lead (d)", "False alarm"],
    [
      ["Volume baseline", "0.28", "0.95", "0.43", "0.49", "6.2", "40%"],
      ["Logistic regression", "0.79", "0.96", "0.87", "0.98", "1.9", "25%"],
      ["Gradient boosting", "0.94", "0.91", "0.93", "0.99", "1.8", "5%"],
      ["GRU", "0.79", "0.98", "0.88", "0.94", "2.0", "20%"],
    ]
  ),
  ...img("fig4_pr_curves.png", 780, 630, "Fig. 3. Precision-recall curves on synthetic data (extracted features)."),
  body("The most important gradient-boosting features are the 2-window rolling count of posts with a trigger (importance 0.77) and the change in on-ground posts (0.10): sustained, rising on-ground activity, not isolated spikes, signals escalation (Fig. 4). Fig. 5 shows the trade-off between warning early and alarming quiet regions."),
  ...img("fig3_feature_importance.png", 975, 540, "Fig. 4. Top features of the gradient-boosting forecaster."),
  ...img("fig5_lead_time.png", 1350, 510, "Fig. 5. Mean days of warning and share of quiet regions with a false alarm, per model."),
  h3("C. Forecasting on Real Data"),
  body("Real escalation is far harder to forecast (Table III, Fig. 6). The best average precision is the GRU's 0.13, about three times the 0.04 base rate; the other models stay near 0.06. Logistic regression warns before 13 of the 19 test onsets, mostly one day ahead, but only 8% of its alarms are correct. Gradient boosting, the best synthetic model, raises almost no alarms at the 0.5 threshold. Replacing our extracted features with GDELT's own CAMEO counts helps only slightly (ROC-AUC 0.70, AP 0.09), which suggests the limit is the predictability of real escalation, not the quality of our extraction."),
  tableCaption("TABLE III. REAL FORECASTING, GDELT INDIA (TEST: SEP–OCT 2026)"),
  metricsTable(
    ["Model (features)", "Prec.", "Recall", "ROC-AUC", "AP", "Onsets warned"],
    [
      ["Volume baseline", "0.05", "0.32", "0.62", "0.06", "11/19"],
      ["Logistic regr. (ours)", "0.08", "0.45", "0.63", "0.06", "13/19"],
      ["Gradient boost. (ours)", "0.00", "0.00", "0.60", "0.06", "0/19"],
      ["GRU (ours)", "0.10", "0.25", "0.56", "0.13", "5/19"],
      ["Logistic regr. (GDELT codes)", "0.08", "0.45", "0.70", "0.09", "14/19"],
      ["GRU (GDELT codes)", "0.07", "0.26", "0.59", "0.07", "5/19"],
    ]
  ),
  ...img("fig6_real_pr_curves.png", 840, 630, "Fig. 6. Precision-recall on the real test period (solid: our headline features; dashed: GDELT codes)."),
  body("Fig. 7 shows why. In Uttar Pradesh, the state with the most onsets, escalations often arrive without a visible build-up in protest coverage, and the headlines classed as escalation rise with, not before, the onsets. News headlines also report events after they happen, while the social posts that might carry earlier signals are the data we could not access at scale."),
  ...img("fig7_real_timeline.png", 1200, 480, "Fig. 7. Uttar Pradesh: protest headlines per day, headlines Stage 1 classes as escalation, and GDELT escalation onsets."),
  h3("D. Sensitivity: Stricter Escalation Threshold"),
  body("Requiring at least three violent-protest articles in a day leaves 24 onsets, only 7 of them in the test period (1.7% of test windows positive). These severe escalations are easier to rank: every model on our headline features reaches ROC-AUC 0.72–0.78, against 0.53 for logistic regression on GDELT's codes, and logistic regression warns before 6 of the 7 test onsets. However, the volume baseline now has the highest average precision (0.24 against 0.16 for logistic regression): large escalations come with a surge of protest coverage, which raw counts already capture. With only 7 test onsets these numbers are very uncertain, but they suggest that extracted events add most value for the smaller escalations that volume misses."),
);

// VII. Discussion
B.push(
  h2("VII. Discussion and Limitations"),
  body("Domain adaptation is the clearest lesson. A model that is near perfect on clean synthetic text loses more than half its F1 on real text, and 80 annotated real items recover most of it; for this task, a small real annotation effort is worth more than a stronger architecture alone."),
  body("The study departs from the project brief in three ways, all forced by data access. (1) Twitter/X, Facebook and Reddit require paid or approved API access, so Mastodon posts stand in for social media. (2) The real forecasting text is news headlines recovered from GDELT URLs, not social posts, because GDELT links events to articles. (3) Only English is handled, although the brief notes that unrest text is often multilingual."),
  body("Other limitations: GDELT's automatic coding labels many off-topic articles as protests, so both the escalation labels and the headline text are noisy, and ACLED's human-coded events would be stronger ground truth; the real annotated set is small and was labelled by one annotator, so real-text scores have wide error bars; and the alarm threshold is fixed at 0.5 rather than tuned."),
  body("Ethics: unrest forecasting can support public-safety and humanitarian preparedness, but the same capability could be misused to surveil or suppress legitimate protest. We collected only public posts from accounts that opted in to indexing, store no user identities, and argue that any deployment must be scoped, governed and audited accordingly."),
);

// VIII. Conclusion
B.push(
  h2("VIII. Conclusion and Future Work"),
  body("We built and evaluated a full pipeline for civil-unrest event extraction and escalation forecasting: a fine-tuned Twitter-RoBERTa extractor for triggers and six argument roles, an escalation-stage classifier, a leakage-free aggregation into region-day windows, and four forecasters including a GRU sequence model. Adapted with 80 real items, the transformer reaches 0.63 span F1 on real text, the best of the four extractors. Forecasting is near perfect on synthetic data but weak on real GDELT data from India (best AP 0.13 against 0.04), at about the level of GDELT's own event codes."),
  body("Future work: obtain ACLED and Reddit/X research access for human-coded ground truth and real social posts; grow the annotated real set to 500+ items with a second annotator; add multilingual support (XLM-R [5]) for Hindi and code-mixed text; calibrate and tune the alarm threshold; and add tone, actor-type and neighbouring-state features, since unrest in one region often precedes unrest nearby."),
);

// References
B.push(
  h2("References"),
  refItem(1, "K. Leetaru and P. A. Schrodt, “GDELT: Global data on events, location, and tone, 1979–2012,” in Proc. ISA Annual Convention, 2013."),
  refItem(2, "C. Raleigh, A. Linke, H. Hegre, and J. Karlsen, “Introducing ACLED: An armed conflict location and event dataset,” Journal of Peace Research, vol. 47, no. 5, pp. 651–660, 2010."),
  refItem(3, "J. Lafferty, A. McCallum, and F. Pereira, “Conditional random fields: Probabilistic models for segmenting and labeling sequence data,” in Proc. ICML, 2001."),
  refItem(4, "D. Q. Nguyen, T. Vu, and A. T. Nguyen, “BERTweet: A pre-trained language model for English tweets,” in Proc. EMNLP: System Demonstrations, 2020."),
  refItem(5, "A. Conneau et al., “Unsupervised cross-lingual representation learning at scale,” in Proc. ACL, 2020."),
  refItem(6, "N. Ramakrishnan et al., “‘Beating the news’ with EMBERS: Forecasting civil unrest using open source indicators,” in Proc. ACM SIGKDD, 2014."),
  refItem(7, "J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” Annals of Statistics, vol. 29, no. 5, 2001."),
  refItem(8, "“Civil Unrest Event Extraction and Prediction from Social Media Data,” Project brief PJ_04, Dept. of Computer Science Engineering, Bennett University, 2026."),
  refItem(9, "Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv:1907.11692, 2019."),
  refItem(10, "F. Barbieri, J. Camacho-Collados, L. Espinosa-Anke, and L. Neves, “TweetEval: Unified benchmark and comparative evaluation for tweet classification,” in Findings of EMNLP, 2020."),
  refItem(11, "P. A. Schrodt, “CAMEO: Conflict and Mediation Event Observations event and actor codebook,” Pennsylvania State University, 2012."),
  refItem(12, "K. Cho et al., “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” in Proc. EMNLP, 2014."),
);

const doc = new Document({
  styles: { default: { document: { run: { font: "Times New Roman" } } } },
  sections: [
    titleSection,
    {
      properties: {
        page: {
          size: { width: convertInchesToTwip(8.5), height: convertInchesToTwip(11) },
          margin: { top: convertInchesToTwip(0.6), bottom: convertInchesToTwip(1.0), left: convertInchesToTwip(0.75), right: convertInchesToTwip(0.75) },
        },
        column: { count: 2, space: 360 },
        type: SectionType.CONTINUOUS,
      },
      children: B,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "civil_unrest_paper.docx"), buf);
  console.log("Wrote civil_unrest_paper.docx");
});
