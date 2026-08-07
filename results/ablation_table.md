# Feature-group ablation

XGBoost trained with fixed (untuned) hyperparameters (n_estimators=200, max_depth=6, learning_rate=0.1), varying only which
feature group is available, on the same time-based split as every other
model (`src/eval/dataset.py`). Threshold fixed at 0.5 (neutral) since this
measures ranking quality (ROC-AUC/PR-AUC), not a tuned operating point.

| Feature group | # features | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| cpu | 5 | 0.8125 | 0.0188 | 0.0565 | 0.0803 | 0.0663 |
| memory | 7 | 0.8268 | 0.0269 | 0.0642 | 0.1334 | 0.0867 |
| disk | 2 | 0.7078 | 0.0071 | 0.0010 | 0.0000 | 0.0001 |
| scheduling | 3 | 0.7847 | 0.0056 | 0.0000 | 0.0000 | 0.0000 |
| all features | 17 | 0.9070 | 0.0510 | 0.1151 | 0.1392 | 0.1260 | **<- full model**

**`memory` alone recovers the most standalone signal among individual groups** (ROC-AUC 0.8268), vs. the full model's 0.9070 -- 91.2% of the full model's ranking quality from one feature group alone.
