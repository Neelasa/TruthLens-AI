# TruthLens AI — Literature Survey

## Research Area

AI-powered misinformation detection, explainable AI, evidence-based
verification, dataset bias, model generalization, source credibility,
LLM-generated misinformation, and multimodal misinformation analysis.

---

## Literature Review Table

| No. | Paper | Year | Dataset | Method / Model | Key Finding | Limitation / Research Issue | Relevance to TruthLens AI |
|---|---|---:|---|---|---|---|---|
| 1 | “Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection | 2017 | LIAR | Hybrid CNN using text and metadata | Introduced a 12.8K manually labelled benchmark of short statements for fake-news detection and showed that combining metadata with text improved over a text-only model. | Focuses on short statements and benchmark classification; does not provide the full evidence-supported, explainable decision-support pipeline envisioned by TruthLens AI. | Provides a strong foundation for the initial text dataset and demonstrates that misinformation can be represented with multiple truthfulness levels. |
| 2 | Hidden Biases in Unreliable News Detection Datasets | 2021 | Multiple unreliable-news datasets | Analysis of dataset and source-overlap effects | Demonstrated that selection bias and source overlap can create misleadingly high performance; accuracy dropped by more than 10% under a clean split without source overlap. | Shows that ordinary random train/test splits may not adequately measure real-world generalization. | Motivates careful dataset splitting, leakage checks, source-overlap analysis, and robust evaluation in TruthLens AI. |
| 3 | Automatic Fake News Detection: Are Models Learning to Reason? | 2021 | Political fact-checking datasets | Claim + evidence analysis | Investigated whether fact-checking models genuinely reason over claims and evidence; found that evidence alone could often produce strong performance. | Indicates that models may exploit properties of evidence rather than genuinely reasoning about the claim. | Supports separating model prediction from evidence-based verification and motivates careful evaluation of the verification component. |
| 4 | Improving Evidence Retrieval for Automated Explainable Fact-Checking | 2021 | Open-domain fact-checking data / web evidence | Quin+; dense retrieval and sentence selection | Proposed a three-stage fact-checking system using evidence retrieval and selection, demonstrating improved evidence recall in noisy web settings. | Web-scale evidence retrieval remains difficult because of noise and the large search space. | Directly supports the future Evidence Retrieval and Verification module of TruthLens AI. |
| 5 | A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection | 2022 | Two explainable fake-news datasets | CofCED neural network | Uses raw reports from multiple media outlets and selects explanatory sentences to generate evidence-based explanations, reducing dependence on already fact-checked reports. | Depends on retrieved reports and still addresses a specialized explainable fake-news setting rather than a complete user decision-support system. | Supports TruthLens AI's Explain and Evidence components and the idea of showing supporting/contradicting evidence. |
| 6 | Explainable Tsetlin Machine Framework for Fake News Detection with Credibility Score Assessment | 2022 | PolitiFact and GossipCop | Tsetlin Machine | Demonstrated an interpretable logic-based approach and credibility assessment; reported competitive results against deep-learning baselines including BERT and XLNet. | The approach is specialized and does not provide multimodal analysis or external evidence verification. | Reinforces the importance of interpretable predictions and credibility/risk scoring rather than unexplained binary output. |
| 7 | Interpretable Multimodal Misinformation Detection with Logic Reasoning | 2023 | Multimodal misinformation data | Multimodal model with logic reasoning | Addresses misinformation containing multiple modalities and emphasizes interpretable reasoning across multimodal content. | Multimodal reasoning introduces additional complexity and remains more difficult than text-only analysis. | Provides research support for TruthLens AI's future image-text and multimodal analysis modules. |
| 8 | Causal Intervention and Counterfactual Reasoning for Multi-modal Fake News Detection | 2023 | Multimodal fake-news datasets | Causal intervention and counterfactual reasoning | Identified psycholinguistic and image-related biases and proposed causal methods to reduce spurious correlations in multimodal detection. | Multimodal models can still suffer from hidden dataset biases and require sophisticated debiasing techniques. | Supports our decision to introduce multimodal capabilities only after establishing a reliable text pipeline and to consider bias/generalization during evaluation. |
| 9 | Adapting Fake News Detection to the Era of Large Language Models | 2024 | Human-written and machine-generated fake/real news | Evaluation of detectors under different generation conditions | Shows that detector performance changes across human-written and machine-generated content and identifies bias toward machine-generated text. | Traditional datasets may not adequately represent the mixture of human- and AI-generated misinformation encountered today. | Provides a future research direction for testing TruthLens AI against AI-generated and paraphrased misinformation. |
| 10 | TELLER: A Trustworthy Framework for Explainable, Generalizable and Controllable Fake News Detection | 2024 | Fake-news datasets | Trustworthy fake-news detection framework | Highlights problems involving non-transparent reasoning, poor generalization, and risks associated with integrating LLMs into fake-news detection. | Trustworthiness requires more than raw classification accuracy and remains challenging when models are opaque or poorly generalized. | Strongly aligns with TruthLens AI's core goal of explainability, generalization, confidence assessment, and trustworthy decision support. |

---

## Emerging Research Themes

### 1. Dataset Bias

Several studies show that misinformation datasets can contain selection bias,
source overlap, and other artifacts that allow models to achieve high
performance without learning the underlying misinformation detection task.

Therefore, TruthLens AI should consider:

- Duplicate detection
- Train/test leakage
- Source overlap
- Appropriate data splitting
- Cross-dataset evaluation where feasible
- Reporting limitations alongside performance

---

### 2. Explainability

Existing research increasingly recognizes that a prediction alone is
insufficient for misinformation analysis.

Useful explanations can include:

- Important textual features
- Evidence sentences
- Supporting or contradicting information
- Model reasoning signals
- Credibility or confidence information

TruthLens AI should therefore provide an explanation layer rather than
returning only a binary classification.

---

### 3. Evidence-Based Verification

A recurring research direction is the use of external evidence to determine
whether a claim is supported or contradicted.

TruthLens AI should distinguish between:

1. Model-based classification
2. Evidence retrieval
3. Evidence-based verification

These should not automatically be treated as the same signal.

A model may consider content suspicious while retrieved evidence may be
insufficient to verify the claim.

---

### 4. Model Generalization

High performance on a single benchmark does not necessarily imply real-world
reliability.

Research indicates that models may learn:

- Dataset-specific artifacts
- Source-specific patterns
- Writing style
- Other spurious correlations

TruthLens AI should therefore evaluate not only standard classification
metrics but also robustness and generalization where practical.

---

### 5. Multimodal Misinformation

Modern misinformation can contain:

- Text
- Images
- Audio
- Video

Research shows that multimodal systems introduce additional opportunities
for reasoning but also additional sources of bias.

Therefore, TruthLens AI will initially focus on text and progressively extend
to image, audio, and video analysis.

---

### 6. AI-Generated Misinformation

The emergence of LLM-generated content creates a new challenge.

A misinformation detector trained primarily on human-written content may behave
differently when evaluating:

- Human-written real content
- Human-written fake content
- Machine-generated real content
- Machine-generated fake content
- Machine-paraphrased content

This should be considered a future robustness experiment for TruthLens AI.

---

## Preliminary Research Observations

The literature indicates several important limitations in existing
misinformation detection systems:

1. Many systems focus primarily on classification.
2. High benchmark accuracy may not imply real-world generalization.
3. Dataset and source biases can artificially inflate performance.
4. Some systems provide limited explanations for their predictions.
5. Evidence retrieval and verification introduce additional challenges.
6. Multimodal systems face additional bias and generalization problems.
7. LLM-generated misinformation creates new detection challenges.
8. A trustworthy system requires more than a single classification score.

---

## Research Gap

The final research gap will be established after the complete literature
comparison and dataset analysis.

The current literature suggests an opportunity to investigate a system that
combines multiple complementary components rather than treating
misinformation detection as a standalone binary classification task.

The proposed direction is to investigate the integration of:

- Text-based misinformation detection
- Explainable predictions
- Confidence/reliability assessment
- Evidence retrieval
- Evidence-based verification
- Privacy-aware processing
- Robust evaluation and generalization analysis
- Progressive multimodal extension

The exact novelty claim will be finalized only after additional literature
and experimental analysis.

---

## Proposed Contribution of TruthLens AI

TruthLens AI proposes a progressive AI-powered decision-support architecture
that moves beyond simple fake/real classification:

Detect → Analyze → Explain → Verify → Guide

The initial implementation will focus on text misinformation detection.
Subsequent modules will investigate explainability and evidence-based
verification, followed by optional multimodal extensions.

The system will be evaluated experimentally rather than assuming that a
specific model such as BERT is automatically superior.

Potential model comparisons include:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- DistilBERT
- BERT
- RoBERTa

The final model selection will be based on actual experimental results,
performance, computational cost, and research relevance.