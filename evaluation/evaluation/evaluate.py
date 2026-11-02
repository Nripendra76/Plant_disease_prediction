"""
Evaluation and Diagnostic Pipeline for Plant Disease Classification.
Computes:
- Top-1 and Top-5 Accuracy
- Precision, Recall, Macro-F1, Weighted-F1
- Multi-class ROC-AUC (One-vs-Rest)
- Full 38-class Confusion Matrix and Confusion Matrix Heatmap
- Top Confusion Pairs (Error Analysis)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset_loader import load_dataset

def evaluate_model_on_test_set(
    model_path: str,
    test_dir: str,
    batch_size: int = config.BATCH_SIZE,
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    Evaluates saved model on independent test set and exports diagnostic metrics and plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading test dataset from {test_dir} (Batch size: {batch_size})...")
    test_ds = load_dataset(test_dir, batch_size=batch_size, shuffle=False, augment=False, cache=False)

    print("Running batch predictions...")
    y_preds = []
    y_trues = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_preds.append(preds)
        y_trues.append(labels.numpy())

    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)

    y_pred_classes = np.argmax(y_preds, axis=1)
    y_true_classes = np.argmax(y_trues, axis=1)

    # 1. Overall Accuracy
    top1_accuracy = np.mean(y_pred_classes == y_true_classes)

    # 2. Top-5 Accuracy
    top5_correct = 0
    for i in range(len(y_true_classes)):
        top5_indices = np.argsort(y_preds[i])[-5:]
        if y_true_classes[i] in top5_indices:
            top5_correct += 1
    top5_accuracy = top5_correct / len(y_true_classes)

    # 3. Macro and Weighted Metrics
    report_dict = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=config.CLASS_NAMES,
        zero_division=0
    )

    # 4. Multi-class ROC-AUC (OvR)
    try:
        roc_auc = roc_auc_score(y_trues, y_preds, multi_class='ovr', average='macro')
    except Exception as e:
        roc_auc = None

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # 6. Error Analysis: Top Confusions
    confusions = []
    for i in range(len(config.CLASS_NAMES)):
        for j in range(len(config.CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'True Class': config.CLASS_NAMES[i],
                    'Predicted Class': config.CLASS_NAMES[j],
                    'Count': int(cm[i, j])
                })
    confusion_df = pd.DataFrame(confusions).sort_values(by='Count', ascending=False)
    confusion_df.to_csv(os.path.join(output_dir, "top_confusions.csv"), index=False)

    # Export Text Report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    # Plot Confusion Matrix
    plt.figure(figsize=(24, 20), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Holdout Test Set)", fontsize=18, fontweight='bold', pad=20)
    plt.colorbar()
    tick_marks = np.arange(len(config.CLASS_NAMES))
    plt.xticks(tick_marks, config.CLASS_NAMES, rotation=90, fontsize=8)
    plt.yticks(tick_marks, config.CLASS_NAMES, fontsize=8)
    plt.xlabel("Predicted Label", fontsize=14, labelpad=10)
    plt.ylabel("True Label", fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    results = {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'macro_precision': report_dict['macro avg']['precision'],
        'macro_recall': report_dict['macro avg']['recall'],
        'macro_f1': report_dict['macro avg']['f1-score'],
        'weighted_f1': report_dict['weighted avg']['f1-score'],
        'roc_auc': roc_auc,
        'report_text': report_text
    }

    print("\n" + "="*50)
    print("EVALUATION SUMMARY:")
    print(f"Top-1 Accuracy:  {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy:  {top5_accuracy:.4f}")
    print(f"Macro F1-Score:  {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.4f}")
    if roc_auc:
        print(f"Macro ROC-AUC:   {roc_auc:.4f}")
    print("="*50)
    print(f"Outputs written to {output_dir}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .keras model")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test images")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory for results")
    args = parser.parse_args()

    evaluate_model_on_test_set(args.model_path, args.test_dir, output_dir=args.output_dir)
