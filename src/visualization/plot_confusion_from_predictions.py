import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ensure plotting works without display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PRED_DIR = 'predictions'
OUT_DIR = os.path.join('results', 'confusion_matrices')
os.makedirs(OUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(PRED_DIR)):
    if not fname.endswith('.npz'):
        continue
    data = np.load(os.path.join(PRED_DIR, fname))
    y_true = data['y_true']
    y_prob = data['y_prob']
    # derive predicted labels from probabilities
    if y_prob.ndim == 2:  # handle multi-class probability matrices
        y_pred = np.argmax(y_prob, axis=1)
    else:  # binary probs shape (n_samples,)
        y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(title=f'{fname.replace(".npz", "")}', xlabel='Predicted label', ylabel='True label', xticks=[0,1], yticks=[0,1])
    # annotate cells with percentage values
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val*100:.2f}%', ha='center', va='center', color='black')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{fname.replace(".npz", "")}_cm.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'Saved {out_path}')
