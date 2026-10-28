"""
Quantum Kernel SVM vs Classical RBF SVM for breast cancer diagnosis
(Wisconsin Diagnostic Breast Cancer dataset, 569 patients, 30 features).

Fair comparison: both models use the SAME splits, SAME scaling, SAME PCA features.
Only the kernel differs (classical RBF vs quantum ZZ feature-map fidelity kernel).

The quantum circuit is simulated exactly with a NumPy statevector simulator
(no external quantum SDK needed), plus a finite-shot simulation to model real hardware noise.
"""
import json, time, itertools
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

SEED = 42
N_QUBITS = 6          # = number of PCA features
REPS = 2              # feature-map repetitions
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
N_REPEATS = 10

# ---------------- Quantum feature map (ZZFeatureMap, Havlicek et al. 2019) ----------------
def zz_state(x, reps=REPS):
    """Statevector of |phi(x)> = [U_Z(x) H^n]^reps |0>, with
    U_Z(x) = exp(i sum_j x_j Z_j + i sum_{j<k} (pi-x_j)(pi-x_k) Z_j Z_k)."""
    n = len(x)
    dim = 2 ** n
    # bit table: z[b, j] = +1/-1 eigenvalue of Z_j on basis state b
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    z = 1 - 2 * bits
    phase = z @ x
    for j, k in itertools.combinations(range(n), 2):
        phase = phase + (np.pi - x[j]) * (np.pi - x[k]) * z[:, j] * z[:, k]
    diag = np.exp(1j * phase)
    psi = np.zeros(dim, complex); psi[0] = 1
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Hn = H
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)
    for _ in range(reps):
        psi = diag * (Hn @ psi)
    return psi

def quantum_kernel(A, B, shots=None, rng=None):
    SA = np.array([zz_state(a) for a in A])
    SB = np.array([zz_state(b) for b in B])
    K = np.abs(SA.conj() @ SB.T) ** 2          # fidelity kernel |<phi(a)|phi(b)>|^2
    if shots is not None:                      # hardware-like finite sampling (compute-uncompute test)
        K = rng.binomial(shots, np.clip(K, 0, 1)) / shots
    return K

def circuit_resources(n=N_QUBITS, reps=REPS):
    pairs = n * (n - 1) // 2
    # per rep: n H + n RZ + per pair (CNOT, RZ, CNOT)
    per_rep_gates = n + n + 3 * pairs
    per_rep_2q = 2 * pairs
    # compute-uncompute kernel circuit = U(x) then U(y)^dagger
    total = 2 * reps * per_rep_gates
    two_q = 2 * reps * per_rep_2q
    depth = 2 * reps * (2 + 3 * pairs)   # conservative (serialised ZZ blocks)
    return dict(qubits=n, reps=reps, gates_per_kernel_circuit=total,
                two_qubit_gates=two_q, depth_upper_bound=depth, trainable_circuit_params=0)

# ---------------- Data ----------------
data = load_breast_cancer()
X = data.data
y = 1 - data.target          # 1 = malignant (positive class), 0 = benign

def prepare(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=N_QUBITS, random_state=SEED).fit(sc.transform(Xtr))
    Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    mm = MinMaxScaler((0, np.pi)).fit(Ztr)
    return np.clip(mm.transform(Ztr), 0, np.pi), np.clip(mm.transform(Zte), 0, np.pi), pca.explained_variance_ratio_.sum()

def metrics(ytrue, ypred, score):
    return dict(accuracy=accuracy_score(ytrue, ypred), recall=recall_score(ytrue, ypred),
                precision=precision_score(ytrue, ypred, zero_division=0), f1=f1_score(ytrue, ypred),
                roc_auc=roc_auc_score(ytrue, score))

def inner_cv(n_train):
    k = 3 if n_train < 40 else 5
    return StratifiedKFold(k, shuffle=True, random_state=SEED)

def fit_classical(Ztr, ytr):
    g = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]},
                     cv=inner_cv(len(ytr)), scoring="roc_auc").fit(Ztr, ytr)
    return g.best_estimator_

BANDWIDTHS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

def fit_quantum(Ztr, ytr):
    """Tune kernel bandwidth beta (features scaled by beta before encoding; Shaydulin & Wild 2022)
    and SVM C with inner CV. Returns (model, beta)."""
    best = (-1, None, None)
    for beta in BANDWIDTHS:
        K = quantum_kernel(beta * Ztr, beta * Ztr)
        g = GridSearchCV(SVC(kernel="precomputed", class_weight="balanced"), {"C": [0.1, 1, 10, 100]},
                         cv=inner_cv(len(ytr)), scoring="roc_auc").fit(K, ytr)
        if g.best_score_ > best[0]:
            best = (g.best_score_, g.best_estimator_, beta)
    return best[1], best[2]

def fit_logreg(Ztr, ytr):
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ztr, ytr)

# ---------------- Experiment 1: full data, 5-fold outer CV ----------------
results = {"full_cv": {"classical_rbf_svm": [], "quantum_kernel_svm": [], "logistic_regression": [],
                       "quantum_kernel_svm_1024_shots": []},
           "learning_curve": {}, "resources": circuit_resources(), "timing": {}}
rng = np.random.default_rng(SEED)
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
t_c = t_q = 0.0
for tr, te in outer.split(X, y):
    Ztr, Zte, ev = prepare(X[tr], X[te])
    t0 = time.time(); clf = fit_classical(Ztr, y[tr]); t_c += time.time() - t0
    results["full_cv"]["classical_rbf_svm"].append(metrics(y[te], clf.predict(Zte), clf.decision_function(Zte)))
    lr = fit_logreg(Ztr, y[tr])
    results["full_cv"]["logistic_regression"].append(metrics(y[te], lr.predict(Zte), lr.decision_function(Zte)))
    t0 = time.time()
    q, beta = fit_quantum(Ztr, y[tr]); t_q += time.time() - t0
    results.setdefault("betas", []).append(beta)
    Kte = quantum_kernel(beta * Zte, beta * Ztr)
    results["full_cv"]["quantum_kernel_svm"].append(metrics(y[te], q.predict(Kte), q.decision_function(Kte)))
    # finite shots (hardware-like)
    Ktr_s = quantum_kernel(beta * Ztr, beta * Ztr, shots=1024, rng=rng); Ktr_s = (Ktr_s + Ktr_s.T) / 2; np.fill_diagonal(Ktr_s, 1)
    Kte_s = quantum_kernel(beta * Zte, beta * Ztr, shots=1024, rng=rng)
    qs = SVC(kernel="precomputed", C=q.C, class_weight="balanced").fit(Ktr_s, y[tr])
    results["full_cv"]["quantum_kernel_svm_1024_shots"].append(metrics(y[te], qs.predict(Kte_s), qs.decision_function(Kte_s)))
results["timing"] = {"classical_fit_seconds_5folds": t_c, "quantum_sim_fit_seconds_5folds": t_q}
results["pca_explained_variance"] = float(ev)

# ---------------- Experiment 2: small-data learning curve ----------------
for frac in FRACTIONS:
    rows = {"classical_rbf_svm": [], "quantum_kernel_svm": []}
    for r in range(N_REPEATS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED + r)
        if frac < 1:
            Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=frac, stratify=ytr, random_state=SEED + r)
        Ztr, Zte, _ = prepare(Xtr, Xte)
        clf = fit_classical(Ztr, ytr)
        rows["classical_rbf_svm"].append(metrics(yte, clf.predict(Zte), clf.decision_function(Zte)))
        q, beta = fit_quantum(Ztr, ytr)
        Kte = quantum_kernel(beta * Zte, beta * Ztr)
        rows["quantum_kernel_svm"].append(metrics(yte, q.predict(Kte), q.decision_function(Kte)))
    results["learning_curve"][str(frac)] = {"n_train": len(ytr), **rows}

# ---------------- Summaries ----------------
def summarise(lst):
    keys = lst[0].keys()
    return {k: (float(np.mean([d[k] for d in lst])), float(np.std([d[k] for d in lst]))) for k in keys}

summary = {"full_cv": {m: summarise(v) for m, v in results["full_cv"].items()},
           "learning_curve": {f: {"n_train": v["n_train"],
                                  "classical_rbf_svm": summarise(v["classical_rbf_svm"]),
                                  "quantum_kernel_svm": summarise(v["quantum_kernel_svm"])}
                              for f, v in results["learning_curve"].items()},
           "resources": results["resources"], "timing": results["timing"],
           "pca_explained_variance": results["pca_explained_variance"], "betas_full_cv": results["betas"]}
n_tr = int(len(y) * 0.8); n_te = len(y) - n_tr
summary["resources"]["kernel_circuits_train"] = n_tr * (n_tr - 1) // 2
summary["resources"]["kernel_circuits_test"] = n_te * n_tr
summary["resources"]["shots_per_circuit"] = 1024
summary["resources"]["total_shots_full_dataset"] = (n_tr * (n_tr - 1) // 2 + n_te * n_tr) * 1024
json.dump(summary, open("results_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
