# CIS 5190 Final Project — News Source Classification

Binary classification of news source (NBC vs Fox) from headline text.

- **Best leaderboard accuracy:** 0.8342 (weighted hybrid ensemble of two TF-IDF + logistic branches)
- **Submission files:** `final_model.py`, `preprocess.py`, `final_model.pt`

## Layout

```
final_model.py              Hybrid ensemble nn.Module used for submission
final_model.pt              Trained weights for final_model.py
preprocess.py               CSV loading, text cleaning, label derivation
generate_report_figures.py  Plots used in the write-up
run_local_evaluation.py     Local accuracy harness

checkpoints/                All other trained .pt branch checkpoints
model_source/               Branch model definitions, training, and experiments
  model_branch_hash_tfidf.py
  model_branch_word_char_tfidf.py
  train_branch_hash_tfidf.py
  train_branch_word_char_tfidf.py
  build_final_model_checkpoint.py
  exp_*.py                  Earlier baselines and side experiments

url_with_headlines.csv      Main dataset
final_headlines.csv         Larger balanced dataset (used for experiments)
```

## Reproducing the final model

From the project root:

```bash
python3 -m model_source.train_branch_hash_tfidf --c 3 --save-path checkpoints/model_tfidf_c3.pt
python3 -m model_source.train_branch_word_char_tfidf --c 4 --save-path checkpoints/model_vocab_char_c4.pt
python3 -m model_source.build_final_model_checkpoint \
  --hash-ckpt checkpoints/model_tfidf_c3.pt \
  --vc-ckpt   checkpoints/model_vocab_char_c4.pt \
  --hash-weight 0.7 --vc-weight 0.3 \
  --save-path final_model.pt
```

## Local evaluation

```bash
python3 run_local_evaluation.py \
  --model final_model.py \
  --preprocess preprocess.py \
  --csv url_with_headlines.csv \
  --weights final_model.pt
```

## Team

Haroun Shah · Qianzhi (Jerry) Zhao · Arvin Krishnan
