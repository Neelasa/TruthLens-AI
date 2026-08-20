# TruthLens AI — Baseline Experiment

## Experiment

TF-IDF + Logistic Regression

## Dataset

LIAR

## Task

Six-class truthfulness classification.

Classes:

- pants-fire
- false
- barely-true
- half-true
- mostly-true
- true

## Input

Statement text only.

Metadata was not used as model input in this baseline.

## Preprocessing

- Convert text to lowercase
- Remove URLs
- Normalize whitespace
- TF-IDF vectorization
- Unigrams and bigrams

## TF-IDF Configuration

- max_features = 20,000
- ngram_range = (1, 2)
- min_df = 2
- max_df = 0.95
- sublinear_tf = True

## Class Imbalance

Logistic Regression used:

class_weight = "balanced"

## Dataset Split

Training:
10,240

Validation:
1,284

Test:
1,267

## Results

Validation Accuracy:
[INSERT RESULT]

Validation Macro F1:
[INSERT RESULT]

Test Accuracy:
[INSERT RESULT]

Test Macro F1:
[INSERT RESULT]

## Observations

The TF-IDF + Logistic Regression model provides the initial
text-only baseline for TruthLens AI.

Future experiments will compare this baseline with stronger
language models.

## Known Dataset Issues

EDA identified:

- 17 duplicate statements within the training set.
- 5 statement overlaps between training and validation.
- 4 statement overlaps between training and test.
- The downloaded files contain 12,791 records, compared with
  12,836 reported in the original LIAR paper.

These issues will be considered when designing subsequent
experiments and interpreting evaluation results.

