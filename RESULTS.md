# CNN Casting Defect Detection — Evaluation Results

## Dataset
- **Total test images:** 715
- **Classes:** `def_front` (defective), `ok_front` (perfect/OK)

## Model Performance on Test Set

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 0.9930 |
| Precision   | 0.9813 |
| Recall      | 1.0000 |
| F1-Score    | 0.9905 |

## Confusion Matrix

|               | Predicted Defect | Predicted OK |
|---------------|-----------------|--------------|
| **True Defect** | 448             | 5            |
| **True OK**     | 0               | 262          |

## Classification Report

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| def_front   | 1.00      | 0.99   | 0.99     | 453     |
| ok_front    | 0.98      | 1.00   | 0.99     | 262     |

- **Macro avg:** 0.99 precision, 0.99 recall, 0.99 F1-score
- **Weighted avg:** 0.99 precision, 0.99 recall, 0.99 F1-score

## Summary

The model correctly classifies all 262 OK (non-defective) samples with zero false negatives. Only 5 defective samples are misclassified as OK out of 453 total defective samples, giving near-perfect performance suitable for industrial inspection automation.
