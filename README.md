# Metaphorical Reasoning Repository

This repository provides tools and resources for metaphorical reasoning, as introduced in our paper: **"Merely Judging Metaphor is Not Enough: Research on Reasonable Metaphor Detection"**, published in Findings of EMNLP 2024.  

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
```bash
python run_clm_no_trainer.py
```

### 2. Obtain Metaphor Reasoning Results with LLMs
Use the following functions in `data_process.py` to generate metaphor reasoning results (the same thing as below):
```python
get_dataset_reason()         # Generate metaphor reasoning for the dataset.
get_final_dataset_reason()   # Finalize the dataset with complete reasoning annotations.
```

### 3. Zero-Shot Reasoning with LLMs
Run zero-shot metaphorical reasoning and evaluate:
```python
Run_LLM_reason_usage_only() 
Run_LLM_reasoning()          
```

Evaluate predictions from LLaMA3 models:
```python
cal_llama3_label_metrics()   # Calculate label prediction metrics only.
cal_llama3_pred_metrics()   
```

### 4. Textual Entailment-Based Similarity Judgments
Evaluate reasoning similarity with textual entailment:
```python
cal_entail_average_evaluate_score()          
cal_entail_average_evaluate_score_wo_usage() # Similarity evaluation without usage context.
```

### 5. LLM-Based Similarity Judgments
Run similarity predictions and calculate evaluation scores:
```python
Run_similiary_pred()                        
Run_similiary_pred_wo_usage()               # Similarity predictions without usage context.
cal_LLMs_average_evaluate_score()           # Calculate average evaluation scores.
```

### 6. Automatic Evaluation
Compute BLEU scores for reasoning quality:
```python
get_BLEU_ave_scores()
```

