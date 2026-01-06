# 🏆 Profitability Relabeling - Guide Rapide

**Objectif**: Briser le plafond de verre (Win Rate 14% → 22-25%)
**Méthode**: Relabeling basé sur PnL futur (pas sur des proxies)

---

## 📊 Situation Actuelle

**IA actuelle**:
- Direction Accuracy: 92% (excellent!)
- Win Rate Trading: **14%** (catastrophique!)
- **Problème**: Proxy Learning Failure

**4 Configs testées**:
- Toutes utilisent des PROXIES (Durée, Volatilité)
- Config 3 (plus agressif): +40% Prédictivité mais -65% PnL
- Config 4 (conservateur): +4% Prédictivité mais -16% PnL

**Conclusion**: Proxies = compromis imparfait qualité vs volume

---

## 🎯 Solution: Profitability Relabeling (Proposition B)

**Principe**:
> "Ne devine pas ce qui est un piège. MESURE-le."

**Algorithme**:
```
Pour chaque STRONG à t:
  Max Return = meilleur exit possible sur 1h
  Si Max Return < 0.2% (frais):
    → Relabeler Force=WEAK (faux positif)
  Sinon:
    → Garder Force=STRONG (valide)
```

**Pourquoi supérieur**:
- ✅ Retire EXACTEMENT les trades perdants
- ✅ IA apprend patterns visuels VRAIS (pas proxies)
- ✅ Validé par littérature ML (Hard Negative Mining, Target Correction)

---

## 🚀 Prochaines Étapes

### 1. **Mettre à Jour Script de Préparation** (~30 min)

**Fichier**: `src/prepare_data_purified_dual_binary.py`
**Ajouter**: Sauvegarde de `prices_*` dans le .npz (nécessaire pour PnL)

**Je le fais ou vous le faites ?**

---

### 2. **Régénérer Datasets** (~5 min)

```bash
python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC
```

---

### 3. **Tester Proposition B** (~10 secondes)

```bash
python tests/test_profitability_relabeling.py --indicator macd --horizon 12
```

**Attendu**:
- ΔWin Rate: +4-5%
- ΔPrédictivité: +50-60%
- ΔPnL Total: -25% (acceptable)
- Trades filtrés: 30-40%

---

### 4. **Si Succès → Relabeling Complet + Réentraînement**

```bash
# Relabeling des 3 datasets
python src/relabel_dataset_profitability.py --assets BTC ETH BNB ADA LTC

# Réentraînement
python src/train.py --data data/prepared/dataset_*_macd_*_relabeled.npz --epochs 50
python src/train.py --data data/prepared/dataset_*_rsi_*_relabeled.npz --epochs 50
python src/train.py --data data/prepared/dataset_*_cci_*_relabeled.npz --epochs 50
```

**Gain attendu IA**: Win Rate 14% → **22-25%** (+8-11%)

---

## 📚 Documentation Complète

- **Guide complet**: `docs/PROFITABILITY_RELABELING_GUIDE.md`
- **Plan d'exécution**: `EXECUTION_PROFITABILITY.md`
- **Prochaines étapes**: `docs/NEXT_STEPS_PROFITABILITY.md`

---

## ❓ Question Immédiate

**Voulez-vous que je modifie `prepare_data_purified_dual_binary.py` pour ajouter la sauvegarde de `prices` ?**

**OU**

**Préférez-vous un script wrapper rapide** (test sans modifier le pipeline, mais plus lent) ?

---

**Recommandation**: Modifier le pipeline (Option A) - propre, réutilisable, gain permanent.

