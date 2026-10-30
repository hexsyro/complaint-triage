
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from scipy.sparse import load_npz
from scipy.special import softmax

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


def ev_scores(m, x_test, y_test):
    proba = m.predict_proba(x_test)
    y_pred = m.classes_[proba.argmax(axis=1)]

    roc_auc=roc_auc_score(y_test, proba, multi_class='ovr', average='macro')

    cr=classification_report(y_test, y_pred, digits=3)

    cm=confusion_matrix(y_test, y_pred, normalize='true')
    return roc_auc, cr, cm

def plot(m, cm):
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='.0%', cmap='Blues',
            vmin=0, vmax=1, linewidths=0.5, linecolor='white',
            xticklabels=m.classes_, yticklabels=m.classes_,
            cbar_kws={'label': 'Share of true class'})

    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title('Where the model gets confused')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    model=joblib.load('models/LinearSVC_model.joblib') #wrapped around CalibratedClassifierCV()
    x_test=load_npz('data/processed/x_test.npz')
    y_test=pd.read_csv('data/processed/y_test.csv').squeeze('columns')

    roc_auc, cr, cm=ev_scores(model, x_test, y_test)
    print()
    print()
    print("=== Here is the LinearCVC model evaluation report ===")
    print()
    print()
    print(f"the ROC-AUC score is: {round(roc_auc,4)}")
    print()
    print(f"the Classification Report is: \n{cr}")
    print()
    print(f"the Confusion Matrix is: \n{cm}")
    print()
    print()
    plot(model, cm)
    print(f"saved confusion matrix visualized chart into 'outputs/figures/confusion_matrix.png'")


if __name__ == '__main__':
    main()