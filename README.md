# MetaMed-RAG: Confidence-Calibrated Medical Meta-Retrieval-Augmented Generation

MetaMed-RAG is a research-grade scaffold for building a **meta-model over heterogeneous medical QA experts** (e.g., MedRAG, ClinicalBERT, PubMedBERT adapters).

The system treats each base model as an expert and learns a confidence-calibrated gating function to decide which expert to trust (or when to abstain).

## Why this is not a toy baseline

This implementation includes components typically expected in an M.Tech thesis-level experimental stack:

- **Structured training data ingestion and schema validation**
- **Per-expert confidence feature engineering** from model and retrieval signals
- **Regularized multinomial stacker** with feature normalization
- **Temperature scaling** for calibration
- **Calibration-aware metrics** (accuracy, multiclass Brier score, ECE)
- **Selective prediction controls** (abstention coverage)

## Architecture

1. **Experts** emit answer + uncertainty/retrieval metadata via `ExpertOutput`
2. **Feature builder** transforms each expert output into confidence features
3. **Meta-model** consumes concatenated expert feature vectors and predicts best expert index
4. **Calibrator** rescales logits using temperature scaling
5. **Inference policy** either selects an expert answer or abstains if confidence is below threshold

## Repository layout

- `metamrag/experts.py` — expert protocol + deterministic mock experts
- `metamrag/confidence.py` — feature vector construction
- `metamrag/data.py` — robust JSONL loading + schema checks
- `metamrag/meta_model.py` — normalized, L2-regularized multinomial linear stacker
- `metamrag/calibration.py` — temperature scaling
- `metamrag/evaluation.py` — calibration-aware evaluation metrics
- `metamrag/pipeline.py` — training, calibration persistence, abstention-aware inference
- `metamrag/cli.py` — train/predict entrypoints

## Data schema (`examples/train.jsonl`)

Per JSONL row:

- `query`: clinical question
- `gold_answer`: canonical answer
- `experts`: list of records containing:
  - `name`
  - `answer`
  - `avg_logprob`
  - `entropy`
  - `retrieval_scores`

## Quickstart

### Train

```bash
python -m metamrag.cli train \
  --train-file examples/train.jsonl \
  --model-out artifacts/meta_model.json \
  --abstain-threshold 0.60
```

Example training output:

- `train_accuracy`
- `train_brier`
- `train_ece`
- `abstention_coverage`
- `calibrated_temperature`

### Predict

```bash
python -m metamrag.cli predict \
  --model-file artifacts/meta_model.json \
  --query "What is first-line treatment for uncomplicated UTI in non-pregnant women?" \
  --abstain-threshold 0.60
```

## Extending to real MedRAG / ClinicalBERT

Implement adapters that satisfy:

```python
class Expert(Protocol):
    name: str
    def predict(self, query: str) -> ExpertOutput: ...
```

Then replace `MockExpert` instances in `build_default_pipeline` with production experts.

## Safety note

This repository is an R&D framework; it is **not a clinical decision support device**. Keep abstention enabled and require clinician oversight for deployment.
