const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  ShadingType, PageOrientation, SectionType, LevelFormat, Numbering,
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
      spacing: { before: 200, after: 60 },
      children: [new ImageRun({ data, transformation: { width: Math.round(pxW * scale), height: Math.round(pxH * scale) }, type: "png" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18 })],
    }),
  ];
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text: text.toUpperCase(), bold: true, size: 20 })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, italics: true, size: 20 })],
  });
}
function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 264 },
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.noIndent ? undefined : { firstLine: 200 },
    children: [new TextRun({ text, size: 20 })],
  });
}
function bullet(text) {
  return new Paragraph({
    spacing: { after: 60 },
    bullet: { level: 0 },
    children: [new TextRun({ text, size: 20 })],
  });
}
function refItem(num, text) {
  return new Paragraph({
    spacing: { after: 80 },
    indent: { left: 260, hanging: 260 },
    children: [new TextRun({ text: `[${num}] `, size: 18 }), new TextRun({ text, size: 18 })],
  });
}

function metricsTable(headers, rows) {
  const mkCell = (t, opts = {}) => new TableCell({
    width: { size: Math.floor(100 / headers.length), type: WidthType.PERCENTAGE },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F3A5F" } : undefined,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: String(t), size: 18, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000" })],
    })],
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({ children: headers.map((hdr) => mkCell(hdr, { header: true })) }),
      ...rows.map((r) => new TableRow({ children: r.map((c) => mkCell(c)) })),
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
      children: [new TextRun({
        text: "Civil Unrest Event Extraction and Prediction from Social Media Data",
        bold: true, size: 32,
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 20 },
      children: [new TextRun({ text: "Project Report", size: 22, italics: true })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 10 },
      children: [new TextRun({ text: "Department of Computer Science Engineering, Bennett University", size: 20 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 10 },
      children: [new TextRun({ text: "Project Instructor: Dr. Pratima Singh  |  Office: N2-WS-030  |  Pratima.singh1@bennett.edu.in", size: 18, italics: true })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: "[Student Name(s) Here]  •  [Roll Number(s) Here]  •  [Date]", size: 18, color: "555555" })],
    }),
  ],
};

// ---------- Body (two columns) ----------
const bodyChildren = [];

// Abstract + keywords (kept narrow, inside the 2-col flow but styled distinctly)
bodyChildren.push(
  h3("Abstract"),
  body(
    "Civil unrest events are increasingly organized, reported, and discussed on social media before and during their occurrence, offering early-warning signals that traditional, newswire-trained event extraction systems fail to capture because they are not adapted to informal, emotionally charged, and deliberately evasive text. This project presents an end-to-end pipeline that (1) extracts civil-unrest-specific event triggers and arguments (actor, location, target, trigger type) from noisy social media posts using a lexicon-informed conditional random field (CRF) sequence tagger, and (2) aggregates the resulting event stream per region and time window to forecast the emergence or escalation of unrest using a gradient-boosted-tree classifier over temporal trend features. We evaluate both stages on a structurally realistic synthetic corpus that encodes a grievance-to-mobilization-to-escalation process, reporting token-level extraction metrics (precision, recall, F1), classification metrics (accuracy, F1, ROC-AUC) for the prediction task, and a lead-time analysis quantifying how many days in advance escalating regions are flagged. The system reaches 0.99 weighted F1 on extraction and 0.98 ROC-AUC on escalation forecasting on the synthetic benchmark, and we discuss what changes we expect, and how to validate, when the pipeline is pointed at real Twitter/X, Reddit, and ACLED/GDELT data.",
    { noIndent: true }
  ),
  new Paragraph({
    spacing: { after: 200 },
    children: [new TextRun({ text: "Keywords: ", bold: true, italics: true, size: 20 }),
      new TextRun({ text: "civil unrest, event extraction, social media mining, conditional random fields, escalation prediction, early warning systems, GDELT, ACLED", italics: true, size: 20 })],
  }),
);

// I. Introduction
bodyChildren.push(
  h2("I. Introduction"),
  body("Protests, riots, strikes, and public demonstrations rarely happen without warning. In the days and hours before large-scale unrest, grievances are voiced, mobilization calls circulate, and on-the-ground reports accumulate on platforms such as Twitter/X and Reddit. This makes social media a uniquely timely — if noisy — sensor for civil unrest, with direct relevance to public safety planning, humanitarian response, and policy monitoring."),
  body("The central difficulty is that social media text differs sharply from the formal, well-edited corpora (news articles, government reports) that most event extraction systems are trained on. Posts are short, informal, code-mixed, emotionally charged, and sometimes deliberately vague to avoid platform moderation. A system that extracts events reliably from such text must be adapted at the vocabulary, feature, and modeling level to this domain — and extraction alone is not the end goal: a useful system must also translate a stream of extracted events into a forecast of what happens next."),
  body("This project addresses both halves of that problem jointly: event extraction adapted to civil-unrest social media text, and a downstream temporal model that consumes the extracted event sequence to forecast escalation, evaluated not only on accuracy but on how early a true escalation is flagged."),
);

// II. Related Work
bodyChildren.push(
  h2("II. Related Work"),
  body("Event extraction has a long history in NLP, from pattern-based approaches on newswire (MUC, ACE) to neural sequence-labeling and, more recently, transformer-based joint trigger-argument extraction. These systems report strong performance on formal text but are known to degrade substantially when applied to social media, where tokenization, named-entity boundaries, and even trigger vocabulary differ from training distributions."),
  body("A separate line of work uses structured event or protest databases — notably GDELT's Global Knowledge Graph and the Armed Conflict Location & Event Data Project (ACLED) — as large-scale, continuously updated ground truth for unrest and conflict events worldwide. These databases are typically derived from news monitoring rather than social media, and are used here as a target label source rather than as the primary text input, since they lag social media chatter by design."),
  body("Social unrest forecasting has also been studied directly from social media streams (e.g., work in the 'Recorded Future' and ICEWS/EMBERS lineage of event-based forecasting systems), generally using engineered or learned features over aggregated post volume, sentiment, and keyword counts per region/time bucket to predict protest occurrence. This project follows that aggregation-then-forecast structure, but couples it to an explicit, interpretable event-extraction stage (rather than bag-of-words volume alone) so that the model's forecast is traceable to structured arguments — who, where, and over what grievance."),
);

// III. Problem Statement (restated)
bodyChildren.push(
  h2("III. Problem Statement"),
  body("Existing event extraction systems, trained on formal corpora, degrade on informal, emotionally charged social media text describing civil unrest, and most systems stop at extraction without modeling how a sequence of extracted events can be used to anticipate escalation. This project addresses the joint problem of (a) extracting civil-unrest event triggers and arguments (actor, location, time, event type, scale) from noisy social text, and (b) using the resulting event sequences to forecast the emergence or escalation of unrest."),
  body("Research Question: How can event extraction from informal social media text be adapted to reliably identify civil-unrest-specific triggers and arguments, and how effectively can the resulting event sequences predict the emergence or escalation of civil unrest?"),
);

// IV. Methodology
bodyChildren.push(
  h2("IV. Methodology"),
  h3("A. System Overview"),
  body("The pipeline has three stages, implemented as independent, swappable modules connected by two fixed dataframe contracts (post-level event frames, and region/time-window aggregates): (1) Event Extraction, (2) Temporal Aggregation, and (3) Escalation Prediction, followed by (4) Evaluation, including a lead-time analysis specific to the forecasting use case."),
  h3("B. Event Extraction"),
  body("Each post is tokenized and tagged with a BIO scheme over four roles: TRIGGER (civil-unrest action words — protest, strike, clash, riot, sit-in, and repression indicators such as tear gas or curfew), ACTOR (who is acting — students, farmers, unions, etc.), LOCATION, and TARGET (the grievance an action is directed at). We use a linear-chain conditional random field (CRF) over a hand-engineered, lexicon-informed feature set: unigram and bigram lexicon membership (to catch multi-word triggers such as \"tear gas\" or \"sit-in\"), word shape and casing (useful for hashtags and ALL-CAPS \"BREAKING\" posts common in social media), and a local context window of neighboring tokens. The CRF is fully trainable offline in seconds on commodity CPU hardware, which made it the right choice for this project's compute environment."),
  body("The project brief specifies a fine-tuned transformer-based extraction module as the target production architecture. We agree this is the stronger approach at scale, and the codebase documents a direct upgrade path: freeze the same (tokens, BIO-labels) training interface and swap in a transformer such as BERTweet or XLM-R (chosen for their fit to noisy/code-mixed social text) fine-tuned for token classification. No other pipeline stage needs to change for this upgrade, since aggregation and prediction consume extracted event frames regardless of which tagger produced them."),
  h3("C. Temporal Aggregation"),
  body("Extracted event frames are bucketed per region into rolling multi-day windows (3 days in the reported experiments). For each window we compute counts of posts at each inferred escalation stage (grievance, mobilization, on-ground report, escalation), the number of distinct actors and targets mentioned, and first-order trend features — the change and rolling mean of each count relative to the previous window(s) — to capture momentum rather than just raw level."),
  h3("D. Escalation Prediction"),
  body("A gradient-boosted decision tree classifier is trained to predict whether a region's next time window will contain escalation-stage activity, given the current and recent windows' aggregated features. Regions are split into train/test sets at the region level (not the window level) so that a region's characteristic baseline chatter cannot leak between training and evaluation — the model must generalize to regions it has not seen rather than memorize a per-region baseline. A from-scratch NumPy GRU sequence model is included in the codebase as an ablation / teaching reference for a fully learned (rather than hand-engineered) temporal encoder."),
  h3("E. Evaluation"),
  body("Extraction is evaluated with token-level precision, recall, and F1 per argument role on a held-out set of posts. Prediction is evaluated with accuracy, precision, recall, F1, and ROC-AUC on held-out regions. Because the practical value of a forecasting system lies in how early it raises an alarm, we additionally report a lead-time analysis: for every region that truly escalates, we identify the first window in which the model's predicted probability crosses the decision threshold and compute the number of days between that flag and the true escalation window."),
);

// V. Dataset
bodyChildren.push(
  h2("V. Dataset"),
  body("Because this project's compute environment has no network path to the Twitter/X API, the Reddit API, GDELT, or ACLED, and no path to download pretrained transformer weights, we built a synthetic-but-structurally-realistic corpus generator that encodes the same process real unrest data exhibits: a region's social chatter progresses from grievance posts, to mobilization calls, to on-ground reports, and — for a subset of regions — to escalation (clashes, repression). The generator samples from civil-unrest trigger vocabulary, actor/location/target gazetteers, and template structures with randomized surface realization, and emits gold BIO labels for the extraction task and ground-truth escalation dates for the prediction task."),
  body("The benchmark used in this report contains 2,192 posts across 16 regions over a 21-day period, with 6 regions constructed to escalate and 10 held flat/noisy as negatives. Figure 1 shows the daily post volume by inferred stage for one escalating region, illustrating the ramp the prediction model is trained to anticipate."),
  ...img("fig1_event_stage_timeline.png", 1125, 600, "Fig. 1. Event-stage escalation timeline for a representative escalating region (synthetic benchmark)."),
  body("We emphasize that extraction scores on this synthetic corpus are an upper bound, not a claim about real-world performance: because posts are template-generated, lexicon-based features can match surface forms almost perfectly. Section VII discusses what we expect to change on real data and how to validate it."),
  body("data/README_DATA_SOURCES.md in the project repository documents, in full, how to replace the synthetic generator with real Twitter/X API v2 and Reddit (PRAW) collection, cross-referenced against GDELT GKG 2.0 and ACLED for ground-truth escalation labeling, without altering any downstream module."),
);

// VI. Results
bodyChildren.push(
  h2("VI. Experimental Setup and Results"),
  h3("A. Extraction Results"),
  body("The CRF tagger was trained on an 80/20 post-level split of the synthetic corpus (1,753 train / 439 test posts). Table I reports per-role token-level metrics on the held-out set."),
  metricsTable(
    ["Role", "Precision", "Recall", "F1"],
    [
      ["B-ACTOR", "1.000", "0.993", "0.997"],
      ["B-TRIGGER", "1.000", "0.994", "0.997"],
      ["B-TARGET", "1.000", "1.000", "1.000"],
      ["I-TARGET", "0.996", "1.000", "0.998"],
      ["B-LOCATION", "1.000", "1.000", "1.000"],
      ["I-TRIGGER", "1.000", "1.000", "1.000"],
      ["Weighted avg", "0.999", "0.998", "0.998"],
    ]
  ),
  new Paragraph({ spacing: { before: 100, after: 160 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "TABLE I. TOKEN-LEVEL EXTRACTION METRICS BY ARGUMENT ROLE", size: 16, bold: true })] }),
  ...img("fig2_extraction_scores.png", 1125, 600, "Fig. 2. Extraction performance by argument role."),
  h3("B. Escalation Prediction Results"),
  body("The escalation classifier was trained on a region-grouped 70/30 split (no region appears in both train and test). On the held-out test regions the model reaches 0.943 accuracy, 0.833 precision, 0.833 recall, 0.833 F1, and 0.980 ROC-AUC, with confusion matrix [[28, 1], [1, 5]] (true negatives, false positives; false negatives, true positives)."),
  ...img("fig4_roc_curve.png", 750, 750, "Fig. 3. ROC curve for escalation forecasting on held-out regions (AUC = 0.98)."),
  body("Figure 4 shows the classifier's top predictive features: the count of on-ground-stage posts in the current window (n_on_ground) dominates, followed by the rolling mean of total post volume and on-ground posts over the previous two windows — consistent with the intuition that sustained on-ground activity, not isolated spikes, is the strongest near-term predictor of escalation."),
  ...img("fig3_feature_importance.png", 1125, 600, "Fig. 4. Top features for escalation forecasting (gradient-boosted trees)."),
  h3("C. Lead-Time Analysis"),
  body("Of the test-set regions that truly escalate, the model detected escalation in advance or at the true window for the majority, with a mean lead time of approximately zero to several days depending on region (Figure 5); one region's escalation was missed entirely at the default 0.5 threshold, indicating headroom for threshold tuning or additional features (e.g., sentiment intensity, sustained hashtag co-occurrence) in future work."),
  ...img("fig5_lead_time.png", 975, 600, "Fig. 5. Early-warning lead time by region on the held-out test set."),
);

// VII. Discussion & Limitations
bodyChildren.push(
  h2("VII. Discussion and Limitations"),
  body("Three limitations bound how far these results generalize. First, the benchmark is synthetic: it is structurally realistic (same stages, same trigger vocabulary, same aggregation process real unrest data would require) but template-generated text inflates extraction scores relative to what a model will see on genuinely noisy, human-written posts — validating on a small hand-labeled real sample is the immediate next step before deployment. Second, the extraction module is a CRF rather than the fine-tuned transformer specified in the original brief, a substitution forced by the absence of network access to pretrained weights in this project's compute environment rather than a methodological preference; Section IV-B documents the exact upgrade path. Third, ground-truth escalation timing in a real deployment would come from ACLED/GDELT, which themselves lag real events by hours to days (they are largely derived from news monitoring) — meaning real-world lead-time numbers will differ from, and are likely more conservative than, the synthetic benchmark's."),
  body("We also note an ethical consideration inherent to this class of system: unrest-forecasting tools can be used for public-safety and humanitarian preparedness, but the same capability could be misapplied for surveillance or suppression of legitimate protest activity. Any deployment should be scoped, governed, and audited accordingly — a point we return to in the conclusion."),
);

// VIII. Conclusion
bodyChildren.push(
  h2("VIII. Conclusion and Future Work"),
  body("This project implemented and evaluated a full pipeline for civil-unrest event extraction and escalation prediction from social media text: a lexicon-informed CRF tagger for triggers and arguments, a temporal aggregation layer, and a gradient-boosted-tree escalation forecaster, evaluated with extraction metrics, prediction metrics, and a lead-time analysis tailored to the early-warning use case. On a structurally realistic synthetic benchmark, the system reaches 0.99 weighted extraction F1 and 0.98 prediction ROC-AUC, and flags the majority of escalating regions at or before the true escalation window."),
  body("Future work should prioritize, in order: (1) validating the extraction module on a small hand-labeled real Twitter/Reddit sample; (2) fine-tuning a transformer tagger (BERTweet / XLM-R) once GPU and model-hub access is available, using the documented upgrade path; (3) replacing synthetic escalation labels with ACLED/GDELT ground truth for a real lead-time evaluation; and (4) extending the prediction model with sentiment intensity and cross-region diffusion features, since unrest in one region frequently correlates with and precedes unrest in neighboring regions."),
);

// References
bodyChildren.push(
  h2("References"),
  refItem(1, "GDELT Project, \"The GDELT Global Knowledge Graph (GKG),\" gdeltproject.org, 2024."),
  refItem(2, "Armed Conflict Location & Event Data Project (ACLED), \"ACLED Codebook,\" acleddata.com, 2024."),
  refItem(3, "J. Lafferty, A. McCallum, and F. Pereira, \"Conditional random fields: Probabilistic models for segmenting and labeling sequence data,\" in Proc. ICML, 2001."),
  refItem(4, "D. Nguyen, T. Vu, and A. T. Nguyen, \"BERTweet: A pre-trained language model for English tweets,\" in Proc. EMNLP: System Demonstrations, 2020."),
  refItem(5, "A. Conneau et al., \"Unsupervised cross-lingual representation learning at scale,\" in Proc. ACL, 2020."),
  refItem(6, "N. Ramakrishnan et al., \"'Beating the news' with EMBERS: Forecasting civil unrest using open source indicators,\" in Proc. ACM SIGKDD, 2014."),
  refItem(7, "J. Friedman, \"Greedy function approximation: A gradient boosting machine,\" Annals of Statistics, 2001."),
  refItem(8, "\"Civil Unrest Event Extraction and Prediction from Social Media Data,\" Project Brief, Dept. of Computer Science Engineering, Bennett University, 2026."),
);

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman" } } },
  },
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
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "civil_unrest_paper.docx"), buf);
  console.log("Wrote civil_unrest_paper.docx");
});
