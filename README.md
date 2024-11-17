---

# Metaphorical Reasoning Repository

This repository provides tools and resources for metaphorical reasoning, including dataset preparation, model training, and evaluation. It supports both supervised and zero-shot approaches using Large Language Models (LLMs).

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yc-cy/Metaphorical-Reasoning.git
cd Metaphorical-Reasoning
```

### 2. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train a Metaphorical Reasoning Model
Train a causal language model for metaphor reasoning:
```bash
python run_clm_no_trainer.py
```

### 2. Obtain Metaphor Reasoning Results with LLMs
Use the following functions in `data_process.py` to generate metaphor reasoning datasets (the same thing as below):
```python
get_dataset_reason()         # Generate metaphor reasoning for the dataset.
get_final_dataset_reason()   # Finalize the dataset with complete reasoning annotations.
```

### 3. Zero-Shot Reasoning with LLMs
Run zero-shot metaphorical reasoning and evaluate:
```python
Run_LLM_reason_usage_only()  # Zero-shot reasoning using usage context.
Run_LLM_reasoning()          # General zero-shot reasoning.
```

Evaluate predictions from LLaMA3 models:
```python
cal_llama3_label_metrics()   # Calculate label prediction metrics.
cal_llama3_pred_metrics()    # Evaluate reasoning prediction quality.
```

### 4. Textual Entailment-Based Similarity Judgments
Evaluate reasoning similarity with textual entailment:
```python
cal_entail_average_evaluate_score()          # Evaluate using entailment metrics.
cal_entail_average_evaluate_score_wo_usage() # Similarity evaluation without usage context.
```

### 5. LLM-Based Similarity Judgments
Run similarity predictions and calculate evaluation scores:
```python
Run_similiary_pred()                        # Generate similarity-based reasoning predictions.
Run_similiary_pred_wo_usage()               # Similarity predictions without usage context.
cal_LLMs_average_evaluate_score()           # Calculate average evaluation scores.
```

### 6. Automatic Evaluation
Compute BLEU scores for reasoning quality:
```python
get_BLEU_ave_scores()
```

---

## License

This project is licensed under the [MIT License](LICENSE).
