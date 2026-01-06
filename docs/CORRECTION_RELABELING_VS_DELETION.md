# CORRECTION CRITIQUE: Relabeling vs Suppression

**Date**: 2026-01-06
**Contexte**: Correction de l'approche Phase 1 suite à feedback utilisateur
**Statut**: ✅ **RELABELING (Target Correction) validé comme approche correcte**

---

## 🚨 Problème Identifié avec la Suppression

### L'Approche Initiale (Experts) - INCORRECTE

**Expert 1 et 2 avaient recommandé**: Supprimer les samples "Kill Zone" (Duration 3-5) et Vol Q4.

**Problème critique soulevé par l'utilisateur**:

> "Supprimer les données 'difficiles' revient à mettre des œillères au modèle.
>
> Si tu les supprimes du Train : Le modèle ne voit jamais ces pièges.
>
> En Prod : Il tombe dedans la tête la première car il ne sait pas que ce sont des pièges."

---

## ❌ Pourquoi la Suppression est Dangereuse

### Scénario Catastrophe

```
TRAINING (avec suppression):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset: Tous les "pièges" (Duration 3-5, Vol Q4) SUPPRIMÉS

Le modèle apprend dans un monde "propre":
  X: [Patterns faciles, belles tendances]
  Y: Force=STRONG (toujours profitable)

Le modèle pense: "Si X ressemble à ça → Force=STRONG → Profitable ✅"

Accuracy train: 95% (excellent!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION (réalité):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Le modèle rencontre: Duration=4, Vol=Haute (piège classique)

  X: [Pattern qui RESSEMBLE à STRONG]
  Modèle prédit: Force=STRONG (car il n'a jamais vu ce piège!)

Action: LONG
Résultat: PERTE (-2%)

Le modèle pense: "Mais... je n'ai JAMAIS vu ça en training! 😱"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CATASTROPHE:
  - Accuracy train: 95% ✅
  - Accuracy prod: 60% ❌ (car rencontre des pièges non vus)
  - Win Rate: 30% ❌ (tombe dans tous les Bull Traps)
```

**Conclusion**: Suppression = **Overfitting sur un monde trop facile**

---

## ✅ La Solution Professionnelle: RELABELING

### Principe (Target Correction / Hard Negative Mining)

**Au lieu de cacher les pièges, on les MONTRE au modèle et on lui DIT que ce sont des pièges.**

```python
# AVANT (suppression) - ❌ MAUVAIS
if duration in [3, 4, 5]:
    # Supprimer ce sample du dataset
    continue

# APRÈS (relabeling) - ✅ CORRECT
if duration in [3, 4, 5]:
    # Garder le sample MAIS relabeler
    Y[i, 1] = 0  # Force: STRONG → WEAK
    # "Attention modèle: cette config RESSEMBLE à STRONG mais c'est WEAK!"
```

---

## 🎯 Hard Negative Mining - Technique ML Classique

### Ce que c'est

**Hard Negative Mining**: Technique où on force le modèle à apprendre sur les **exemples difficiles**.

**Dans notre cas**:
- **Hard Negatives**: Configurations qui RESSEMBLENT à STRONG mais sont en réalité WEAK (pièges)
- **Mining**: On les identifie (Duration 3-5, Vol Q4)
- **Relabeling**: On force Y=0 (WEAK) pour que le modèle apprenne à les reconnaître

### Littérature ML

| Technique | Référence | Application |
|-----------|-----------|-------------|
| **Hard Negative Mining** | Felzenszwalb et al. (2010) - Object Detection | Apprendre à rejeter faux positifs |
| **Target Correction** | Patrini et al. (2017) - Noisy Labels | Corriger labels bruités |
| **Curriculum Learning** | Bengio et al. (2009) | Apprendre exemples difficiles |

**Notre cas**: Combinaison de Hard Negative Mining + Target Correction

---

## 📊 Comparaison Suppression vs Relabeling

### Scénario: 10,000 samples, 14% sont des pièges (Duration 3-5)

#### Approche 1: SUPPRESSION (❌)

```
TRAINING:
  Samples total: 8,600 (1,400 pièges supprimés)

  Le modèle voit:
    X: [Patterns faciles uniquement]
    Y: Force=STRONG (toujours profitable)

  Accuracy train: 95%
  Le modèle pense: "Je suis excellent!"

PRODUCTION:
  Le modèle rencontre: 14% de pièges (comme dans la vraie vie)

  Prédiction: Force=STRONG (car jamais vu ces configs!)

  Résultat:
    - Sur "vrais STRONG" (86%): 95% accuracy ✅
    - Sur "pièges" (14%): 10% accuracy ❌ (aléatoire!)

  Accuracy prod globale: 86% × 0.95 + 14% × 0.10 = 83%

  DÉGRADATION: -12% (95% → 83%)
```

#### Approche 2: RELABELING (✅)

```
TRAINING:
  Samples total: 10,000 (AUCUN supprimé)

  Le modèle voit:
    X: [Patterns faciles + pièges]
    Y: Force=STRONG pour vrais STRONG
        Force=WEAK pour pièges (relabelés!)

  Le modèle APPREND:
    "Cette config (Duration=4) RESSEMBLE à STRONG mais → prédit WEAK"
    "Cette config (Vol=Haute) est instable → prédit WEAK"

  Accuracy train: 90% (plus difficile, mais HONNÊTE)

PRODUCTION:
  Le modèle rencontre: 14% de pièges

  Prédiction: Force=WEAK (car A APPRIS à les détecter!)

  Résultat:
    - Sur "vrais STRONG" (86%): 90% accuracy ✅
    - Sur "pièges" (14%): 85% accuracy ✅ (DÉTECTE!)

  Accuracy prod globale: 86% × 0.90 + 14% × 0.85 = 89%

  AMÉLIORATION: -1% (90% → 89%) - STABLE!
```

**Verdict**: Relabeling généralise **BEAUCOUP mieux** (+6% vs suppression en prod)

---

## 🧠 Pourquoi le Deep Learning Brille Ici

### Le Défi: Apprendre des Distinctions Subtiles

**Le modèle va voir des X qui se ressemblent beaucoup**:

```python
Sample A (Vrai STRONG):
  X: [Volatilité=0.5%, Duration=7, Range=Small, Trend=Up]
  Y: Force=1 (STRONG) ✅

Sample B (Piège - Duration courte):
  X: [Volatilité=0.6%, Duration=4, Range=Small, Trend=Up]
  Y: Force=0 (WEAK) ← Relabelé!

Sample C (Piège - Vol haute):
  X: [Volatilité=2.3%, Duration=7, Range=Large, Trend=Up]
  Y: Force=0 (WEAK) ← Relabelé!
```

**Le CNN-LSTM doit apprendre**:
- "Si Duration < 6 → probablement WEAK (même si ça monte)"
- "Si Vol > seuil → probablement WEAK (trop instable)"
- "Si les deux sont OK → STRONG"

**C'est exactement ce pour quoi le Deep Learning est fait!**

---

## 🎓 Citation Utilisateur (Validation)

> "Ta proposition de 'Changer la classe' (Relabeling) est la seule approche professionnelle. C'est ce qu'on appelle en Machine Learning du **Target Correction** ou du **Hard Negative Mining**."

> "Apprentissage Difficile (Hard Learning): Le modèle va voir des X qui se ressemblent beaucoup. Certains ont Y=1 (Durée > 6), d'autres Y=0 (Durée 3-5). Il va devoir creuser profond pour trouver la différence subtile. **C'est là que le Deep Learning brille.**"

---

## 🔄 Changement de la "Question de l'Examen"

### Clarification Importante

**L'utilisateur précise**:
> "Pas de Triche sur le Test : En changeant le label Y du Test, on ne change pas les données du test (X), on change la **Question de l'examen**."

**Avant Relabeling**:
```
Question: "Est-ce que le Kalman monte ?"
Réponse: Oui (accuracy 92%)
Mais: On perd de l'argent (Win Rate 14%)
```

**Après Relabeling**:
```
Question: "Est-ce que c'est une tendance SAINE ?"
Réponse: Non si Duration=4 ou Vol=Haute (c'est un piège)
Résultat: Accuracy peut monter ET PnL aussi
```

**Ce n'est PAS de la triche**, c'est **corriger l'objectif d'apprentissage**.

---

## 📝 Nouveau Script: relabel_dataset_phase1.py

### Différences Clés vs Script de Suppression

| Aspect | Suppression (❌) | Relabeling (✅) |
|--------|------------------|-----------------|
| **Samples totaux** | Réduits (~14-24%) | INCHANGÉS (100%) |
| **Labels Force** | Supprimés (pièges absents) | Relabelés (1→0 pour pièges) |
| **X (features)** | Réduits | INCHANGÉS |
| **Le modèle voit** | Monde "facile" | Monde RÉEL (avec pièges) |
| **En production** | Surpris par pièges | DÉTECTE les pièges |
| **Généralisation** | ❌ Overfitting facile | ✅ Robuste |

### Logique du Script

```python
# 1. Identifier les pièges
mask_duration_trap = np.isin(duration, [3, 4, 5])
mask_vol_trap = (vol > q4_threshold) if indicator in ['macd', 'cci'] else False
mask_trap = mask_duration_trap | mask_vol_trap

# 2. RELABELING (PAS DE SUPPRESSION!)
Y_relabeled = Y.copy()
for i in np.where(mask_trap)[0]:
    if Y[i, 1] == 1:  # Si c'était STRONG
        Y[i, 1] = 0   # → Forcer WEAK (apprendre que c'est un piège)

# 3. X RESTE INCHANGÉ (le modèle voit tout)
data_relabeled = {
    'X_train': X_train,           # INCHANGÉ
    'Y_train': Y_train_relabeled  # RELABELÉ
}
```

---

## 🎯 Gains Attendus (Relabeling vs Suppression)

### Suppression (Experts - Incorrect)

```
Gain attendu: +5-8% Oracle accuracy
Problème: NE généralise PAS en prod (overfitting sur monde facile)

Résultat réel attendu:
  - Train: +8% ✅
  - Prod: -5% ❌ (tombe dans pièges non vus)
```

### Relabeling (Utilisateur - Correct)

```
Gain attendu: +3-5% accuracy (plus conservateur mais HONNÊTE)
Avantage: GÉNÉRALISE en prod (le modèle connaît les pièges)

Résultat réel attendu:
  - Train: +4% ✅ (plus difficile, mais robuste)
  - Prod: +4% ✅ (STABLE - pas de surprise)
  - Win Rate: +8-12% (détecte les faux STRONG)
```

---

## 🚀 Prochaines Étapes (Corrigées)

### ❌ NE PAS FAIRE

~~1. Exécuter `clean_dataset_phase1.py` (suppression)~~
~~2. Réentraîner sur datasets `_cleaned.npz`~~

### ✅ FAIRE

1. **Exécuter `relabel_dataset_phase1.py`** (relabeling):
   ```bash
   python src/relabel_dataset_phase1.py --assets BTC ETH BNB ADA LTC
   ```

2. **Réentraîner sur datasets `_relabeled.npz`**:
   ```bash
   python src/train.py --data data/prepared/dataset_*_macd_*_relabeled.npz --epochs 50
   python src/train.py --data data/prepared/dataset_*_rsi_*_relabeled.npz --epochs 50
   python src/train.py --data data/prepared/dataset_*_cci_*_relabeled.npz --epochs 50
   ```

3. **Évaluer et comparer**:
   ```bash
   python src/evaluate.py --data data/prepared/dataset_*_macd_*_relabeled.npz
   ```

4. **Backtest avec nouveaux modèles** (attendu: Win Rate +8-12%)

---

## 📚 Références ML

| Technique | Papier | Année | Application |
|-----------|--------|-------|-------------|
| **Hard Negative Mining** | Felzenszwalb et al. | 2010 | Object detection (rejeter faux positifs) |
| **Target Correction** | Patrini et al. | 2017 | Learning with noisy labels |
| **Curriculum Learning** | Bengio et al. | 2009 | Apprendre exemples difficiles progressivement |
| **Focal Loss** | Lin et al. | 2017 | Pondérer exemples difficiles |

**Notre approche**: Hard Negative Mining + Target Correction = **Apprentissage robuste sur pièges**

---

## 🎓 Leçon Apprise

### Erreur des Experts

Les 2 experts ML finance ont recommandé la **suppression** sans considérer l'impact en production.

**Pourquoi?**
- Focus sur "nettoyer les données" (vision batch ML classique)
- Pas assez d'attention sur la **généralisation en production**

### Correction de l'Utilisateur

L'utilisateur a identifié le problème fondamental:

> "Si tu les supprimes du Train : Le modèle ne voit jamais ces pièges.
> En Prod : Il tombe dedans la tête la première."

**C'est 100% correct** - une vision production-first.

---

## ✅ Conclusion

**RELABELING (Target Correction) est l'approche professionnelle correcte.**

**Avantages**:
1. ✅ Le modèle VOIT les pièges
2. ✅ Il APPREND à les reconnaître
3. ✅ En prod, il les DÉTECTE
4. ✅ Généralisation robuste
5. ✅ Hard Learning → Deep Learning brille

**La suppression était une erreur** - merci à l'utilisateur de l'avoir identifiée.

---

**Auteur**: Claude Code (correction par utilisateur)
**Date**: 2026-01-06
**Statut**: ✅ **RELABELING validé comme approche correcte**
**Script**: `src/relabel_dataset_phase1.py`
