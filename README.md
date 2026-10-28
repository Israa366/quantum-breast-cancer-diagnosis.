Quantum Kernel SVM vs Classical SVM for Breast Cancer Diagnosis

Author: Israa Mazen Alasaad, University of Sharjah (BSc Cybersecurity Engineering)

Question: Can a quantum kernel diagnose breast cancer as well as a tuned classical model, under a fair comparison, and what does it cost in quantum resources?

1. The healthcare problem
Problem: classify a breast tumour as benign or malignant from fine-needle aspirate (biopsy) measurements.
Relevance: breast cancer is the most common cancer in women. Early and reliable diagnosis reduces missed cancers and unnecessary procedures.
Data: Wisconsin Diagnostic Breast Cancer dataset (Wolberg, Street & Mangasarian, UCI Machine Learning Repository, loaded via scikit-learn). It has 569 patients and 30 numeric features: 212 malignant and 357 benign.
Intended outcome: high recall (sensitivity) on the malignant class, meaning few missed cancers, and high ROC-AUC, using a compact model small enough for current quantum devices.
2. Shared pipeline (same for both models)

To keep the comparison fair, both models use exactly the same steps. Only the kernel is different.

Standardise the features (z-score). The scaler is fitted on training folds only, so there is no data leakage.
Reduce the features with PCA to 6 components, which keep 88.6% of the variance. These 6 features map to 6 qubits.
Min-max scale the features to [0, π], the angle range used for encoding.
Use stratified 5-fold outer cross-validation. Hyperparameters are tuned by inner cross-validation on ROC-AUC, and every model uses class_weight="balanced".
The malignant class is the positive class.
3. Classical baseline
Model: scikit-learn SVC with an RBF kernel.
Grid: C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, 0.01, 0.1, 1}.
Extra reference: logistic regression on the same features.
Why this baseline: the quantum model is also a kernel SVM, so an RBF SVM isolates the effect of the kernel itself. SVMs are a well-established method on this dataset (Bennett & Mangasarian, and the standard scikit-learn benchmarks), and we credit that prior work.
4. Quantum contribution
Model: a quantum kernel SVM. The RBF kernel is replaced by a quantum fidelity kernel inside the same SVM solver.
Feature map: a ZZ feature map (Havlíček et al., Nature 567, 2019) on 6 qubits with 2 repetitions. Each repetition applies Hadamard gates, then single-qubit Z rotations by xᵢ, then entangling ZZ rotations by (π − xᵢ)(π − xⱼ) for every pair of qubits.
Kernel: k(x, x′) = |⟨φ(x)|φ(x′)⟩|².
Bandwidth: the features are multiplied by a bandwidth β before encoding, with β ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 1.0}, tuned together with C by inner cross-validation. This follows Shaydulin & Wild, Phys. Rev. A 106, 042407 (2022).
Why a quantum kernel: the entangling ZZ gates encode pairwise feature interactions in a 2⁶ = 64-dimensional Hilbert space. The kernel has no trainable circuit parameters, so it avoids the barren-plateau training problems of variational quantum models, and kernel methods suit small medical datasets.
Simulation: an exact NumPy statevector simulator, plus a 1,024-shot finite-sampling simulation that mimics measurement noise on real hardware.
Key finding: kernel concentration

The naive kernel (β = 1) failed, with ROC-AUC 0.82 and only 15% recall. At full-strength encoding the quantum states are almost orthogonal, so the kernel matrix is close to the identity matrix (exponential concentration), and every patient looks equally different from every other. Tuning the bandwidth fixed this: β = 0.1 was selected in all 5 folds, and ROC-AUC rose to 0.994.

How this follows "Introduction to Practical Quantum Computing"
Resource concept	Where it is used in this project
Qubits, superposition (§1)	6 qubits; Hadamard gates put every qubit in superposition
Rotation encoding (§2.2, §3.1)	Each PCA feature is scaled to [0, π] and encoded as a phase rotation P(2x)
Multi-qubit gates, entanglement (§2.3)	CNOT–P–CNOT blocks entangle every pair of qubits to capture feature relationships
Hybrid algorithm (§4)	The quantum circuit computes the kernel; a classical SVM trains and predicts
Quantum feature maps + SVM (§5.4)	Data is mapped into a 2⁶ = 64-dimensional space, where the SVM finds a separating boundary
Shots (§5.5.2)	1,024-shot simulation to model measurement sampling

We chose the quantum feature map route instead of a quantum neural network (§5.5), because it has no trainable circuit parameters and needs no classical optimiser loop, so training is more stable on a small medical dataset.

qiskit_version.py builds the same model with Qiskit's ZZFeatureMap and FidelityStatevectorKernel. We checked numerically that our NumPy simulator and Qiskit's circuit convention give the same kernel (maximum difference about 10⁻¹⁵).

5. Results
Full dataset (5-fold CV, mean)
Model	Accuracy	Recall	Precision	F1	ROC-AUC
Classical RBF SVM	0.970	0.967	0.955	0.960	0.996
Quantum kernel SVM (exact)	0.963	0.967	0.942	0.952	0.994
Quantum kernel SVM (1,024 shots)	0.953	0.948	0.930	0.937	0.992
Logistic regression (reference)	0.967	0.944	0.967	0.954	0.995
Quantum kernel SVM, naive (β = 1)	0.677	0.151	0.358	0.211	0.822

The fold-to-fold standard deviation of recall is ±0.024 for both models, and the quantum model's precision varies by ±0.063. The differences in accuracy and precision are smaller than this variation.

Small-data learning curve (test ROC-AUC, mean of 10 random splits, fixed 30% test set)
Training patients	Classical RBF SVM	Quantum kernel SVM
19	0.967	0.731
39	0.980	0.894
99	0.988	0.971
199	0.993	0.985
398	0.994	0.991

We tested whether the quantum kernel generalises better from very few patients. It did not. The classical model is clearly better at 19 and 39 patients, and the gap closes as the amount of data grows.

Quantum resource and efficiency metrics
Metric	Value
Qubits	6
Feature-map repetitions	2
Trainable circuit parameters	0
Gates per kernel circuit (compute–uncompute)	228 (120 two-qubit)
Circuit depth (upper bound)	≤ 188
Kernel circuits, full dataset (train + test)	103,285 + 51,870 = 155,155
Shots at 1,024 per circuit	≈ 159 million
Fit time over 5 folds (simulator)	6.3 s quantum vs 1.6 s classical
6. Advantages, trade-offs and limitations

Advantages

The same malignant recall as the classical model (96.7%), so the quantum model misses no extra cancers.
No trainable circuit parameters, so there are no barren plateaus and no variational training.
Robust to shot noise: AUC is still 0.992 at 1,024 shots.
Only 6 qubits, which fits on today's quantum devices.

Trade-offs and limitations

No accuracy advantage, and precision is slightly lower (by about 1.3 points).
Weaker with very small training sets.
The number of kernel circuits grows with N², which is costly on real hardware.
The model was only simulated; real device noise was not tested.
PCA to 6 features keeps 88.6% of the variance, so some information is lost.

We do not claim a quantum advantage. The contribution is reaching clinical parity on recall with a parameter-free, noise-robust 6-qubit kernel, and showing that bandwidth tuning is essential for quantum kernels on real medical data.

7. How to run
bash
pip install -r requirements.txt
python qkernel_breast_cancer.py

The script runs in about 50 seconds on a laptop, prints all results, and writes results_summary.json.

Qiskit version (single train/test split, β = 0.1):

bash
pip install qiskit qiskit-machine-learning scikit-learn
python qiskit_version.py
8. Files
qkernel_breast_cancer.py: full pipeline, classical baseline, quantum kernel simulator and all experiments
qiskit_version.py: the same quantum kernel built with Qiskit (ZZFeatureMap + FidelityStatevectorKernel)
results_summary.json: all results (means and standard deviations), resource metrics and timing
requirements.txt: dependencies
9. Future work
Run the kernel on real quantum hardware with error mitigation.
Use kernel approximation (for example, Nyström or random sampling) to reduce the N² circuit cost.
Try other feature maps and more qubits with less PCA reduction.
References
W. H. Wolberg, W. N. Street, O. L. Mangasarian. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository, 1995.
V. Havlíček et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019).
R. Shaydulin, S. M. Wild. Importance of kernel bandwidth in quantum machine learning. Phys. Rev. A 106, 042407 (2022).
F. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830 (2011).
