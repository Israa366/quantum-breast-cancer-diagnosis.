"""
Qiskit version of the quantum kernel (same model as qkernel_breast_cancer.py).

It uses Qiskit's built-in ZZFeatureMap and FidelityStatevectorKernel, following the
"Quantum Feature Maps" approach (Section 5.4 of "Introduction to Practical Quantum Computing").

Install:  pip install qiskit qiskit-machine-learning scikit-learn
Run:      python qiskit_version.py

Note: our NumPy simulator in qkernel_breast_cancer.py builds the same circuit. Its phases are
the complex conjugate of Qiskit's convention, which gives an identical kernel |<phi(x)|phi(x')>|^2.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

N_QUBITS, REPS, BETA, C = 6, 2, 0.1, 10   # BETA = bandwidth chosen by cross-validation in the main script

X, t = load_breast_cancer(return_X_y=True)
y = 1 - t                                   # 1 = malignant
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

sc = StandardScaler().fit(Xtr)
pca = PCA(N_QUBITS, random_state=42).fit(sc.transform(Xtr))
mm = MinMaxScaler((0, np.pi)).fit(pca.transform(sc.transform(Xtr)))
f = lambda A: BETA * np.clip(mm.transform(pca.transform(sc.transform(A))), 0, np.pi)
Ztr, Zte = f(Xtr), f(Xte)

# Rotation encoding + entangling ZZ gates (H, P(2x), CX-P(2(pi-x_i)(pi-x_j))-CX), 2 repetitions
feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=REPS, entanglement="full")
print(feature_map.decompose().draw(fold=120))
qc = feature_map.decompose()
print("qubits:", qc.num_qubits, "| depth:", qc.depth(), "| gates:", dict(qc.count_ops()))

kernel = FidelityStatevectorKernel(feature_map=feature_map)
Ktr = kernel.evaluate(Ztr)
Kte = kernel.evaluate(Zte, Ztr)

svm = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, ytr)
pred, score = svm.predict(Kte), svm.decision_function(Kte)
print(f"Qiskit quantum kernel SVM  accuracy={accuracy_score(yte, pred):.3f}  "
      f"recall={recall_score(yte, pred):.3f}  ROC-AUC={roc_auc_score(yte, score):.3f}")
