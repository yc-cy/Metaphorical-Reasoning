## Repository Overview

This repository includes scripts and resources to:
1. Train supervised models for metaphorical reasoning.
2. Use LLM to obtain metaphor reasoning results.
3. Perform zero-shot reasoning using LLMs.
4. Evaluate reasoning using textual entailment and LLM-based similarity metrics.
5. Automatically evaluate generated reasoning using BLEU scores.

---

## Installation

1. Clone this repository:  
   ```bash
   git clone https://github.com/yc-cy/Metaphorical-Reasoning.git
   cd Metaphorical-Reasoning
   ```

2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Train a Metaphorical Reasoning Model
Train a language model for metaphor reasoning using the following command:
```bash
cd GPT
python run_clm_no_trainer.py
```

### 2. Obtain Metaphor Reasoning Results with LLMs
  ```python
  get_dataset_reason()
  get_final_dataset_reason()
  ```

### 3. Zero-Shot Reasoning with LLMs
- Generate metaphorical reasoning with zero-shot LLMs:  
  ```python
  Run_LLM_reason_usage_only()
  Run_LLM_reasoning()
  ```

- Calculate metrics for LLaMA3 predictions:  
  ```python
  cal_llama3_label_metrics()
  cal_llama3_pred_metrics()
  ```

### 4. Textual Entailment-Based Similarity Judgments
  ```python
  cal_entail_average_evaluate_score()
  cal_entail_average_evaluate_score_wo_usage()
  ```

### 5. LLM-Based Similarity Judgments
- Run similarity-based reasoning predictions:  
  ```python
  Run_similiary_pred()
  Run_similiary_pred_wo_usage()
  ```

- Calculate average evaluation scores with LLMs:  
  ```python
  cal_LLMs_average_evaluate_score()
  ```

### 6. Automatic Evaluation
  ```python
  get_BLEU_ave_scores()
  ```

---

## License
This project is licensed under the [MIT License](LICENSE).  

For questions or contributions, feel free to open an issue or submit a pull request.
