from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate(y_true, y_pred, model_name: str):

    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred,
          target_names=['non-toxic','toxic'], digits=4))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                xticklabels=['non-toxic','toxic'],
                yticklabels=['non-toxic','toxic'])
    
    ax.set_title(f'{model_name} — normalized confusion matrix')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower().replace(" ","_")}_cm.png', dpi=150)
    plt.show()

    return {
        'model': model_name,
        'f1_toxic': f1_score(y_true, y_pred, pos_label=1),
        'precision_toxic': precision_score(y_true, y_pred, pos_label=1),
        'recall_toxic': recall_score(y_true, y_pred, pos_label=1),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }