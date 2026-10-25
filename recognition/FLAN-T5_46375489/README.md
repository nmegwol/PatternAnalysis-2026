# [Project Title]

*Provide a descriptive title identifying the problem and implemented model. Replace the instructions below with your own supported report content.*

## 1. Problem Description and Project Overview

*Describe the problem, intended users, prototype purpose, and scope in approximately one paragraph. Identify the chosen model and the task of translating technical radiology reports into accessible layperson summaries.*

### Open Research Question

*Explain the Task 2.6 dilemma: whether adaptation improves clinical faithfulness or mainly linguistic fluency and ROUGE. State the question investigated, including hallucinations, critical omissions, diagnostic uncertainty, and readability.*

## 2. Algorithm and Working Principles

*Explain how the algorithm works in approximately one paragraph. Describe the model architecture, input-to-output process, and relevant course concepts, supported by literature citations.*

### Architecture or Pipeline Figure

*Include at least one labelled figure or visualisation explaining the model or processing pipeline, with a caption and discussion in the text.*

## 3. One-Page Feasibility Review

*Include the written feasibility review presented for the compulsory check-off. Keep the review to one page and clearly identify the following four required parts.*

### User Need, Scope and Acceptance Criteria

*State the intended user and prototype purpose. Provide 3–5 testable acceptance criteria addressing the engineering dilemma, including an operational/resource constraint such as peak VRAM or runtime.*

### Model Choice and Course Concepts

*Describe the proposed dataset configuration, model architecture, and chosen difficulty band. Explain how course concepts and literature support the choices.*

### Preliminary Feasibility Evidence

*Present an initial data audit, baseline smoke-test experiment, or measured resource estimate. Report actual evidence and its implications; a fully trained final model is not required for this review.*

### Risks, Budget and Fallback

*Identify key technical risks, the computing budget, the next planned experiment, and a workable fallback.*

## 4. Dataset and Preprocessing

### Dataset Source and Configuration

*Identify and cite the dataset and track. Describe the input and target fields, selected configuration, and any subsets used. Provide acquisition instructions without committing dataset files.*

### Training, Validation and Test Protocol

*Describe and justify the split usage, subset selection, and measures taken to prevent leakage. Identify the shared validation split for baseline/model benchmarking and the test cases reserved for qualitative analysis.*

### Preprocessing, Tokenisation and Prompts

*Document cleaning or transformations, tokenisation, source/target length limits, truncation, padding, target loss masking, and task prompts. Explain any augmentation used and cite adapted preprocessing methods.*

## 5. Implementation and Usage

### Code Organisation

*Explain the responsibilities and interactions of modules.py, dataset.py, train.py, and predict.py, and any supporting files actually used.*

### Dependencies and Environment

*List required dependencies with exact versions, installation instructions, and relevant hardware/software details. Document any teaching-staff approval actually obtained for otherwise restricted dependencies.*

### Data Preparation

*Provide runnable commands and required configuration for obtaining and preparing the data, including expected paths and outputs.*

### Training, Validation and Model Saving

*Provide commands and configuration for training, validation, testing where applicable, and saving checkpoints locally. Explain how to run the implemented adaptation modes.*

### Evaluation

*Provide commands for reproducing baseline comparisons, ROUGE measurements, and resource profiling, with the required inputs and expected output locations.*

### Prediction from Saved Weights

*Provide commands for loading saved weights and generating summaries for test examples. Explain checkpoint paths, input format, and output format without committing model files.*

## 6. Experimental Design and Reproducibility

### Implemented Baseline

*Describe the baseline actually implemented, its checkpoint or method, prompts, and generation settings. Explain how it supports a fair comparison with the adapted model.*

### LoRA and Full Fine-Tuning

*Describe the adaptation approaches investigated, including hyperparameters, trainable components, optimisation, training budget, and checkpoint selection.*

### Controlled Comparison and Reproduction Settings

*Record seeds, dataset selections, configurations, decoding/stopping criteria, and evaluation settings. Explain how identical evaluation splits and consistent conditions support reproducibility and fair comparison.*

## 7. Quantitative Results and Resource Profiling

### Validation-Set Baseline Comparison

*Present actual baseline and adapted-model results on the same held-out validation split, including ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum. Explain the scoring procedure and interpret the measured differences.*

### Training Loss and Metric Plots

*Include labelled plots of training losses and tracked metrics, explaining convergence, validation behaviour, and relevant differences between adaptation methods.*

### Computational Resource Comparison

*Provide a profiling table reporting hardware/GPU, total and trainable parameter counts, peak GPU VRAM, training time, inference runtime/latency, and inference throughput in tokens/second. State measurement conditions and discuss quality–resource trade-offs.*

## 8. Example Inputs and Outputs

*Show representative input reports, reference summaries where available, and actual generated outputs from the baseline and adapted models. Include relevant visual plots and explain what the examples demonstrate.*

## 9. Clinical Safety Audit and Failure Analysis

### Selection of 3–5 Representative Test Cases

*Explain the selection of 3–5 actual representative failure cases from the test data. Identify cases sufficiently to connect the analysis to the observed outputs.*

### Case-by-Case Error Analysis

*For each case, compare the source report and generated summaries. Analyse hallucinations, critical omissions, changes in diagnostic certainty, and layperson readability. Diagnose likely root triggers and failure modes using the observed evidence.*

### Why ROUGE Does Not Establish Clinical Safety

*Relate the qualitative findings to the quantitative results. Explain why lexical overlap can miss clinically consequential errors and discuss the trade-off between readability and faithful preservation of findings.*

## 10. Engineering Recommendation and Limitations

### Evidence-Based Recommendation and Trade-Offs

*Give the project manager an actionable recommendation supported by baseline comparisons, resource measurements, and error analysis. Explain the benefits and costs of the investigated approaches.*

### Clinical Guardrails and Deployment Risks

*Specify safeguards needed before patient-facing deployment, such as entity verification or confidence filtering where justified. Explain the risks they address and the evidence still needed to assess their effectiveness.*

### Limitations

*State the limits of the data, experiments, metrics, qualitative audit, and generalisation claims. Distinguish measured findings from assumptions and proposed future work.*

## 11. Artificial Intelligence Usage Disclosure

### AI Tools Used

*Identify the AI tools actually used and model/version information where known.*

### Specific Tasks Assisted by AI

*Describe the specific tasks each tool assisted with and which code, explanations, or report material were generated or modified. Reconcile this account with the maintained AI-use log.*

### Audit, Verification and Empirical Testing

*Describe how AI suggestions were audited, verified, corrected, and empirically tested. Distinguish automated or assistant checks from your own review, and identify anything not yet verified.*

## 12. References and Attribution

*Provide citations for the dataset, model literature, evaluation and adaptation methods, and any reused or adapted third-party code or approaches. Connect citations to the relevant report sections and identify adaptations clearly.*

