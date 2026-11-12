# Guideline – Challenge di classificazione (Intro ML)

Checklist da seguire in ordine. Ogni fase ha **cosa fare**, **cosa guardare** e **cosa scrivere nel report**.

> Regola d'oro: **tutto ciò che impara qualcosa dai dati** (scaler, trasformazioni, soglie per gli outlier, PCA, scelta degli iperparametri) si fitta **solo sul training**, meglio dentro una `Pipeline`. Il test si usa **una sola volta**, alla fine.

---

## 0. Setup

- [ ] Import in testa al notebook, una sola volta
- [ ] Fissare un seed globale (`random_state=42` ovunque: split, CV, modelli)
- [ ] Un solo `X`, un solo `y`, **un solo split**: non ridefinire le variabili più avanti

---

## 1. Pretreatment

### 1a. Controlli che NON imparano dai dati (prima dello split)

- [ ] **Struttura**: `df.shape`, `df.dtypes`, `df.head()`
- [ ] **Target**: valori ammessi (`df["class"].isin([0, 1]).all()`)
- [ ] **Valori nulli**: `df.isna().sum()` → decidere se imputare o rimuovere
- [ ] **Duplicati**: `df.duplicated().sum()` → **rimuoverli prima dello split** (altrimenti una copia va in train e una in test e il test diventa più facile)
- [ ] **Ordinamento**: le righe sono ordinate per classe?
  - Problema: split o `KFold` senza shuffle producono fold con una sola classe
  - Soluzione: `stratify=y` nello split e `StratifiedKFold(shuffle=True)` nella CV
- [ ] **Bilanciamento**: `y.value_counts(normalize=True)` → l'accuracy basta come metrica o serve F1?
- [ ] **Separare feature e target**: `X = df.drop(columns="class")`, `y = df["class"]`

### 1b. Split

- [ ] Rispettare la consegna (es. test di **372** elementi):

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=372, stratify=y, shuffle=True, random_state=42)
```

- [ ] Verificare che le proporzioni delle classi siano uguali in train e test

### 1c. Analisi e decisioni (solo su `X_train`)

- [ ] **Distribuzioni**: `X_train.describe()`, istogrammi, `X_train.skew()`
  - $|\text{skew}| < 0.5$ ok · $0.5$–$1$ moderata · $> 1$ forte
- [ ] **Outlier** (regola IQR, fence $[Q_1 - 1.5\,\text{IQR},\ Q_3 + 1.5\,\text{IQR}]$)
  - Percentuale per feature, non solo conteggio
  - Si concentrano in una classe? → allora portano informazione: **non eliminarli**
  - Feature molto skewed → la regola IQR flagga troppo: sono code, non errori
  - Decisione: tenere / trasformare (Yeo-Johnson) / clippare con i bound del training. Il test **non si pulisce mai**
- [ ] **Correlazioni**: `X_train.corr()` → serve per giudicare Naive Bayes
- [ ] **Scaling: serve? perché?**

| Modello | Scaling | Motivo |
|---|---|---|
| k-NN | obbligatorio | la distanza è dominata dalla feature con scala maggiore |
| Logistic regolarizzata | obbligatorio | la penalità $\lambda\|w\|^2$ dipende dalle unità di misura |
| PCA | obbligatorio | altrimenti PC1 segue la feature con varianza più grande |
| Decision tree | inutile | split invarianti a trasformazioni monotone |
| Gaussian NB | inutile | stima media e varianza per ogni feature |

- [ ] Scaler e trasformazioni **dentro la pipeline**, mai `fit` su tutto il dataset

**Nel report**: rispondere esplicitamente alle domande della consegna (scaling sì/no e perché, dati ordinati sì/no, che problema crea, come si risolve).

---

## 2. Unsupervised (esplorazione)

Scopo: **prevedere** come andranno i modelli supervised.

- [ ] **Standardizzare** prima della PCA
- [ ] **PCA**: scree plot, varianza spiegata cumulata, scatter PC1 vs PC2 colorato per classe
  - Domanda: le classi sono linearmente separabili in questa proiezione?
- [ ] **k-means ($k=2$)** sulle 2 PC e su tutte le feature
  - Confrontare i cluster con le classi (confusion matrix cluster vs classe, o ARI)
  - Ricordare che le etichette dei cluster sono arbitrarie (0/1 possono essere invertite)
- [ ] **t-SNE**: struttura non lineare? (le distanze tra cluster in t-SNE non sono interpretabili)
- [ ] **DBSCAN**: scegliere `eps` (k-distance plot) e `min_samples`; contare punti di rumore

**Nel report**: annotare le previsioni (es. "frontiera non lineare → kNN/albero dovrebbero battere la logistic") da confrontare al punto 4.

---

## 3. Modelli supervised

### 3a. Schema uguale per ogni modello

1. [ ] **Baseline**: `DummyClassifier(strategy="most_frequent")`, il numero da battere
2. [ ] **Pipeline**: (scaler) + modello
3. [ ] **`GridSearchCV`** con `StratifiedKFold(5, shuffle=True)`, **solo sul training**
4. [ ] **Curva** score di CV (media ± std) e score di training al variare dell'iperparametro → bias-variance
5. [ ] Modello migliore rifittato su tutto il training (`refit=True`, default)
6. [ ] Valutazione sul test **una volta sola**, alla fine

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1",
                    return_train_score=True)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

### 3b. Cosa guardare per ciascun modello

**Logistic Regression**
- [ ] Iperparametro `C` $= 1/\lambda$ su griglia logaritmica ($10^{-4}$ … $10^{4}$)
- [ ] Confronto penalità `l2` vs `l1` (`solver="liblinear"` o `"saga"`)
- [ ] Effetto della regolarizzazione: $C$ piccolo → alto bias, coefficienti verso 0
- [ ] Coefficienti: con L1 alcuni vanno a 0 → feature selection
- [ ] Limite: frontiera lineare

**Decision Tree (ID3)**
- [ ] `criterion="entropy"` (information gain, come ID3)
- [ ] Regolarizzazione: `max_depth`, `min_samples_leaf`, `ccp_alpha` (pruning)
- [ ] Albero pieno → training accuracy 100% → overfitting
- [ ] Nota per il report: sklearn usa CART (split binari su soglie), non ID3 puro (split multipli su feature categoriche)

**Naive Bayes (`GaussianNB`)**
- [ ] Nessuna vera regolarizzazione (solo `var_smoothing`): dirlo esplicitamente
- [ ] Assunzione: $p(x \mid y) = \prod_j p(x_j \mid y)$, indipendenza condizionale
- [ ] Collegare alla matrice di correlazione: feature molto correlate → assunzione violata
- [ ] Ogni $p(x_j \mid y)$ è gaussiana → feature skewed peggiorano il fit (Yeo-Johnson può aiutare)

**k-NN**
- [ ] Iperparametri: `n_neighbors` (dispari), `weights` (`uniform` / `distance`)
- [ ] Regolarizzazione = $k$: $k=1$ alta varianza, $k$ grande alto bias
- [ ] Scaling obbligatorio
- [ ] Con poche feature il curse of dimensionality non pesa

---

## 4. Valutazione e confronto

- [ ] Metriche sul test + **confusion matrix** per ogni modello

$$\text{Accuracy}=\frac{TP+TN}{n}\qquad \text{Precision}=\frac{TP}{TP+FP}\qquad \text{Recall}=\frac{TP}{TP+FN}\qquad F_1=\frac{2PR}{P+R}$$

- [ ] Dichiarare **quale classe è positiva** e quale errore costa di più (es. banconota falsa accettata = falso negativo → privilegiare la recall)
- [ ] **Tabella riassuntiva**: baseline + modelli × (CV media ± std, accuracy, precision, recall, F1 sul test)
- [ ] Test coerente con la CV? Se il test è molto meglio/peggio → sospettare leakage o split sfortunato
- [ ] **Differenze piccole non sono significative**: con 372 esempi, 1 errore ≈ 0.27% di accuracy
- [ ] Confronto con l'unsupervised: le previsioni del punto 2 sono confermate?
- [ ] **Miglioramenti possibili**: feature polinomiali o kernel per la logistic, ensemble (random forest, boosting), trasformazioni per NB

---

## 5. Controllo finale prima di consegnare

- [ ] Il notebook gira da capo a fondo (`Restart & Run All`) senza errori
- [ ] Un solo split, usato da tutti i modelli
- [ ] Nessun `fit` / `fit_transform` su dati che includono il test
- [ ] Test usato solo nella sezione 4
- [ ] Ogni domanda della consegna ha una risposta scritta in markdown
- [ ] Grafici con titolo, assi etichettati e legenda
- [ ] Baseline presente

---

## Domande tipiche della commissione su questo tipo di progetto

- Hai normalizzato prima o dopo lo split? Perché?
- Dov'è la baseline?
- Perché hai scelto quella metrica?
- Cosa succede se elimini gli outlier concentrati in una classe?
- L'assunzione di Naive Bayes è rispettata dai tuoi dati?
- Che effetto ha combinare PCA e Naive Bayes?
- Quali sono i punti deboli dei dati e del metodo?
- Come scalerebbe il tuo approccio con molti più dati o feature?