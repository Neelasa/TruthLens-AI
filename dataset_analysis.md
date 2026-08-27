# TruthLens AI — Dataset Analysis

## 1. Purpose

The initial version of TruthLens AI will focus on text-based misinformation
analysis.

The dataset should therefore support:

- Supervised text classification
- Reliable labels
- Reproducible train/validation/test evaluation
- Explainability experiments
- Classical ML baselines
- Transformer-based experiments
- Future comparison/generalization studies

---

# 2. Candidate Dataset Comparison

| Dataset | Approx. Size | Content | Labels | Main Strength | Main Concern | Initial Suitability |
|---|---:|---|---|---|---|---|
| LIAR | 12,836 | Short factual statements | 6 truthfulness levels | Human-labelled fact-checking data and standard splits | Short political statements; English only | HIGH |
| ISOT Fake News | 44,898 | Full news articles | Fake / Real | Large and easy binary classification setup | Real and fake articles originate from different source distributions | MEDIUM |
| FakeNewsNet | Multiple collections | News + social context | Fake / Real | News content plus social-context information | Complete dataset cannot simply be redistributed; collection can require external APIs/data | MEDIUM |
| NELA-GT-2018 | ~713,000 | News articles | Source-level ground-truth ratings | Very large and rich source-level information | Source-level labels can introduce publisher/source bias | MEDIUM |
| FEVER | 185,445 claims | Claims + evidence | Supported / Refuted / Not Enough Info | Excellent for later claim verification | Designed for fact verification rather than general fake-news classification | HIGH for future verification |

---

# 3. LIAR Dataset

## Dataset Name

LIAR

## Original Research

“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection

William Yang Wang, 2017.

## Source

PolitiFact.com

## Approximate Size

12,836 manually labelled short statements.

## Language

English.

## Labels

The original LIAR dataset contains six truthfulness categories:

1. pants-fire
2. false
3. barely-true
4. half-true
5. mostly-true
6. true

The six-class formulation preserves more information than a simple
fake/real classification.

## Standard Splits

The commonly distributed dataset contains:

- Training: 10,269
- Validation: 1,284
- Test: 1,283

Total: 12,836.

## Available Features

The dataset contains fields including:

- id
- label
- statement
- subject
- speaker
- job title
- state information
- party affiliation
- previous truthfulness counts
- context

The main input for our first experiment will be the statement text.

Metadata will initially be kept separate so that we can measure the
performance of text-only models without accidentally relying on speaker or
source-related information.

---

# 4. Why LIAR Is Attractive for TruthLens AI

LIAR is a strong candidate for the first experiment because:

1. It was specifically introduced for fake-news detection and fact-checking.
2. It contains human-labelled statements.
3. It provides six levels of truthfulness.
4. It has predefined training, validation and test splits.
5. It is small enough for a single student to experiment with.
6. It is suitable for classical ML and transformer experiments.
7. Its fine-grained labels fit the idea that misinformation is not always
   simply fake or real.

---

# 5. Limitations of LIAR

LIAR should not be treated as a complete representation of real-world
misinformation.

Important limitations include:

### 5.1 Short Statements

The dataset primarily contains short statements rather than complete
long-form news articles.

### 5.2 Political Context

The statements originate from PolitiFact and are strongly associated with
political fact-checking.

### 5.3 English Language

The initial dataset is English-only.

### 5.4 Limited Modality

The dataset is primarily textual and does not provide the multimodal
text-image-audio-video environment targeted by the long-term TruthLens AI
vision.

### 5.5 Generalization

Good performance on LIAR alone should not be interpreted as proof that a model
will perform equally well on unseen real-world misinformation.

---

# 6. ISOT Fake News Dataset

ISOT contains approximately 44,898 articles:

- 21,417 truthful articles
- 23,481 fake articles

The dataset contains article title, text, publication date and subject.

The truthful articles were collected from Reuters, while fake articles were
collected from multiple sources associated with unreliable/fact-checked
content.

## Advantages

- Much larger than LIAR.
- Full news articles.
- Simple binary fake/real task.
- Convenient for classical ML experiments.

## Concerns

The difference in source distributions between the truthful and fake
collections can introduce dataset-specific patterns.

Therefore, a model may potentially learn source or writing-style signals
rather than genuinely learning misinformation characteristics.

## Decision

ISOT will NOT be the primary dataset for our first experiment.

It may be useful later for cross-dataset/generalization evaluation.

---

# 7. FakeNewsNet

FakeNewsNet provides fake and real news collections from PolitiFact and
GossipCop and includes news content and social-context information.

Its repository provides minimal CSV files containing information such as:

- article ID
- URL
- title
- tweet IDs

The complete dataset cannot simply be redistributed because of Twitter
privacy policies and news publisher copyrights.

## Advantages

- News content
- Social-media context
- Multiple sources
- Useful for future multimodal/social misinformation research

## Concerns

- More complex than required for the initial experiment.
- Some information requires additional collection.
- Twitter/API-related data acquisition introduces additional practical
  complications.

## Decision

FakeNewsNet will be considered a future/advanced dataset rather than the
initial dataset.

---

# 8. NELA-GT-2018

NELA-GT-2018 contains approximately 713,000 articles collected from 194
news/media outlets.

It includes source-level ground-truth ratings covering dimensions such as:

- reliability
- bias
- transparency
- journalistic standards
- consumer trust

## Advantages

- Very large dataset.
- Multiple news/media sources.
- Rich source-level information.
- Useful for large-scale misinformation research.

## Concerns

Because ground truth is associated with sources/outlets, models may learn
publisher-specific patterns.

This makes it less suitable as the simplest first dataset for TruthLens AI.

## Decision

NELA-GT may be considered later for robustness/generalization research.

---

# 9. FEVER

FEVER is a fact-verification dataset rather than a conventional fake-news
classification dataset.

It contains claims and evidence and supports labels such as:

- Supported
- Refuted
- Not Enough Information

## Importance to TruthLens AI

FEVER is highly relevant to our future Evidence Verification component.

TruthLens AI eventually needs to distinguish:

Model prediction
from
Evidence-based verification.

Therefore, FEVER may become useful during the future verification stage.

## Decision

FEVER will not be the first classification dataset.

It will be considered for the future claim-verification module.

---

# 10. Final Dataset Decision

## Primary Dataset

### LIAR

LIAR will be used for the initial text misinformation experiment.

## Reason

The decision is based on:

- Appropriate task relevance
- Human-labelled data
- Fine-grained truthfulness labels
- Standard train/validation/test splits
- Manageable size
- Suitability for classical ML
- Suitability for transformer experiments
- Research relevance
- Practical feasibility for an individual student

---

# 11. Experimental Strategy

The initial experiment will use the statement text as the primary input.

We will first establish:

## Baseline

TF-IDF + Logistic Regression

Then compare against appropriate alternatives such as:

- TF-IDF + Linear SVM
- DistilBERT
- BERT
- RoBERTa

The final model will not be selected in advance.

It will be selected based on actual experimental results.

---

# 12. Evaluation Metrics

For the initial classification experiments we will report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For the six-class LIAR task, macro-F1 will be particularly useful because it
provides a class-balanced view of performance.

Where appropriate, additional metrics may be used.

---

# 13. Research Considerations

The following issues will be explicitly considered:

- Data leakage
- Duplicate statements
- Class distribution
- Source/speaker effects
- Dataset bias
- Train/test separation
- Generalization
- Computational cost

The goal is not simply to maximize accuracy.

The goal is to determine whether the model learns useful misinformation-related
signals.

---

# 14. Future Dataset Strategy

After the initial LIAR experiments, we may consider an additional dataset
for cross-dataset evaluation.

Possible candidates:

- ISOT
- FakeNewsNet
- NELA-GT

The final choice will depend on:

- Accessibility
- Licensing/usage conditions
- Data quality
- Compatibility with the first experiment
- Research value

---

# 15. Dataset Selection Status

Status:

PRIMARY DATASET SELECTED

Dataset:

LIAR

Next step:

Download/access the dataset and perform exploratory data analysis before
training any model.