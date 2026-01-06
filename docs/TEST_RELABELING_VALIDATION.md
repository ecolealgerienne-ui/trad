# Test de Validation: Impact du Relabeling

**Date**: 2026-01-06
**Objectif**: Valider que le relabeling améliore Oracle **AVANT** de réentraîner
**Durée**: ~10 secondes
**Script**: `tests/test_relabeling_impact.py`

---

## Principe du Test

### Pourquoi ce Test?

**Problème**: Les experts recommandaient la suppression, nous avons corrigé vers relabeling.
**Question**: Est-ce que relabeler les "pièges" en WEAK améliore réellement Oracle?

**Test rapide AVANT réentraînement**:
1. Charger données **test uniquement** (out-of-sample)
2. Relabeling **en mémoire** (pas de sauvegarde)
3. Comparer Oracle AVANT vs APRÈS relabeling
4. Si amélioration → GO réentraînement ✅
5. Sinon → Revoir approche ❌

---

## Ce que le Script Teste

### 3 Scénarios Comparés

**1. Oracle AVANT (baseline)**:
- Labels originaux (Y_test original)
- Performance: ce qu'on a actuellement

**2. Oracle APRÈS (relabeled)**:
- Labels avec pièges relabelés Force 1→0
- Performance: ce qu'on aurait si l'IA apprenait correctement

**3. IA (référence)**:
- Prédictions actuelles (Y_test_pred)
- Pour comparaison (ne change pas)

---

## Métriques Calculées

### Pour chaque scénario:

**1. Accuracy**:
- Direction Accuracy (UP/DOWN)
- Force Accuracy (WEAK/STRONG)

**2. Prédictivité**:
- Correlation Direction avec returns futurs
- Correlation STRONG avec returns futurs
- **Plus haut = meilleur signal**

**3. Trading Simulé**:
- Logique: LONG si Direction=UP + Force=STRONG
- Métriques:
  - Nombre de trades
  - Win Rate
  - PnL Total
  - Profit Factor

---

## Résultats Attendus

### Hypothèse: Oracle APRÈS > Oracle AVANT

**Si le relabeling est correct**:

```
Oracle AVANT:
   Win Rate:     49.87%
   PnL Total:    +66.44%
   Trades:       ~40,000

Oracle APRÈS (relabeled):
   Win Rate:     55-60% (+5-10%) ✅
   PnL Total:    +75-85% (+10-20%) ✅
   Trades:       ~30,000 (-25%) ✅ (moins de pièges!)

ΔWin Rate:   +5-10%
ΔPnL:        +10-20%
ΔTrades:     -25% (meilleure sélectivité)

→ VALIDATION POSITIVE ✅
```

**Si Win Rate et PnL montent** → Le relabeling est valide
**Si Trades baissent en plus** → Bonus (on filtre les mauvais)

---

## Commandes d'Exécution

### Test MACD (recommandé - meilleur indicateur)

```bash
python tests/test_relabeling_impact.py --indicator macd
```

### Test sur tous les indicateurs

```bash
for ind in macd rsi cci; do
    echo "Testing $ind..."
    python tests/test_relabeling_impact.py --indicator $ind
done
```

---

## Interprétation des Résultats

### ✅ Cas 1: Validation Positive

```
Oracle APRÈS:
   ΔWin Rate:   +8.2%
   ΔPnL:        +15.3%
   ΔTrades:     -9,800

✅ VALIDATION POSITIVE
   → GO pour réentraînement avec datasets relabelés
```

**Action**: Exécuter `relabel_dataset_phase1.py` puis réentraîner

---

### ⚠️ Cas 2: Validation Mitigée

```
Oracle APRÈS:
   ΔWin Rate:   +1.2%
   ΔPnL:        -2.1%
   ΔTrades:     -15,000

⚠️  VALIDATION MITIGÉE
   → Revoir seuils ou approche
```

**Action**: Analyser quels pièges relabelés ne sont pas vraiment des pièges

---

### ❌ Cas 3: Validation Négative

```
Oracle APRÈS:
   ΔWin Rate:   -3.5%
   ΔPnL:        -8.2%
   ΔTrades:     -20,000

❌ VALIDATION NÉGATIVE
   → L'approche relabeling ne fonctionne pas
```

**Action**: Revoir complètement l'approche

---

## Ce que le Test NE Fait PAS

**Important**: Ce test compare **Oracle AVANT vs APRÈS**, pas IA.

**Ce qu'on teste**:
- ✅ Les "pièges" identifiés sont-ils réellement des mauvais trades?
- ✅ Relabeler ces pièges en WEAK améliore-t-il Oracle?
- ✅ Est-ce que ça vaut le coup de réentraîner?

**Ce qu'on NE teste PAS**:
- ❌ Si l'IA va apprendre correctement après réentraînement
- ❌ La nouvelle performance de l'IA (besoin de réentraîner pour ça)

**L'IA restera mauvaise dans ce test** (elle utilise Y_test_pred qui sont les anciennes prédictions).

---

## Logique de Validation

### Pourquoi comparer Oracle AVANT vs APRÈS?

**Oracle = Limite théorique supérieure**

Si on relabele les pièges et que Oracle NE S'AMÉLIORE PAS:
→ Soit les "pièges" ne sont pas vraiment des pièges
→ Soit notre identification est mauvaise

Si Oracle S'AMÉLIORE après relabeling:
→ Les pièges identifiés sont bien des pièges
→ Relabeler est la bonne approche
→ L'IA pourra apprendre à les détecter après réentraînement

---

## Scénario Idéal (Attendu)

```
=================================================================
📊 COMPARAISON ORACLE AVANT vs APRÈS
=================================================================

Oracle AVANT (baseline):
   Direction Accuracy: 100.00%
   Force Accuracy:     100.00%

   Trading Simulé:
     Win Rate:         49.87%
     PnL Total:        +66.44%
     Trades:           38,542

Oracle APRÈS (relabeled):
   Direction Accuracy: 100.00%
   Force Accuracy:     100.00%

   Trading Simulé:
     Win Rate:         58.23% ✅ (+8.36%)
     PnL Total:        +82.71% ✅ (+16.27%)
     Trades:           28,741 ✅ (-9,801 pièges filtrés)

=================================================================
🎯 SYNTHÈSE
=================================================================

Impact Relabeling:
   ΔWin Rate:   +8.36%
   ΔPnL Total:  +16.27%
   ΔTrades:     -9,801

✅ VALIDATION POSITIVE: Relabeling améliore Oracle
   → GO pour réentraînement avec datasets relabelés
```

---

## Prochaines Étapes selon Résultat

### Si Validation Positive ✅

1. **Relabeling complet**:
   ```bash
   python src/relabel_dataset_phase1.py --assets BTC ETH BNB ADA LTC
   ```

2. **Réentraînement** (3 indicateurs):
   ```bash
   python src/train.py --data data/prepared/dataset_*_macd_*_relabeled.npz --epochs 50
   python src/train.py --data data/prepared/dataset_*_rsi_*_relabeled.npz --epochs 50
   python src/train.py --data data/prepared/dataset_*_cci_*_relabeled.npz --epochs 50
   ```

3. **Évaluation** (attendu: IA apprend à détecter les pièges):
   ```bash
   python src/evaluate.py --data data/prepared/dataset_*_macd_*_relabeled.npz
   ```

---

### Si Validation Mitigée ⚠️

**Analyser les résultats**:
- Quels types de pièges ont été mal identifiés?
- Ajuster les seuils (Duration, Vol Q4)?
- Tester d'autres critères?

**Actions possibles**:
- Relabeler uniquement Duration 3-5 (universel validé 100%)
- Ignorer Vol Q4 (pattern moins stable)
- Affiner les seuils

---

### Si Validation Négative ❌

**Revoir l'approche complète**:
- Les "pièges" identifiés ne sont peut-être pas des pièges
- Le Data Audit a peut-être trouvé des corrélations accidentelles
- Retour à la table de dessin

---

## Avantages de ce Test

**1. Rapide** (~10 secondes):
- Pas besoin de réentraîner
- Test immédiat de l'hypothèse

**2. Validant**:
- Compare Oracle AVANT vs APRÈS
- Métriques claires (Win Rate, PnL)

**3. Non destructif**:
- Relabeling en mémoire
- Aucune modification des fichiers

**4. Décisionnel**:
- Résultat clair: GO ou NO-GO
- Évite de perdre du temps si l'approche ne fonctionne pas

---

## Conclusion

Ce test est **crucial** avant de lancer le réentraînement complet.

**10 secondes de test peuvent sauver 3 heures de réentraînement inutile.**

Si Oracle s'améliore après relabeling → L'approche est validée ✅
Sinon → On évite une erreur coûteuse ❌

---

**Auteur**: Claude Code
**Date**: 2026-01-06
**Statut**: Script prêt à l'emploi
**Durée estimée**: 10 secondes par indicateur
