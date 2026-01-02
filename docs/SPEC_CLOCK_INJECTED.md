# Architecture "Clock-Injected" (5min + Step Index)

**Date**: 2026-01-02
**Statut**: Specification (a implementer)
**Version**: 1.0

---

## Vue d'Ensemble

Architecture hybride combinant la reactivite des indicateurs 5min avec la stabilite des indicateurs 30min, synchronisee par un "Step Index" qui indique la position dans la fenetre 30min.

---

## 1. Structure de l'Input X (7 Features)

Le modele recoit une sequence de 12 timesteps. Chaque timestep est compose de 7 colonnes normalisees :

### Indicateurs 5min (3 colonnes)
- **RSI(14)** - Synchronise (Lag 0, Concordance 82%)
- **CCI(20)** - Synchronise (Lag 0, Concordance 74%)
- **MACD(10/26/9)** - Synchronise (Lag 0, Concordance 70%)

> Note: BOL retire car impossible a synchroniser (toujours Lag +1).

### Base 30min Fixe (3 colonnes)
Les valeurs RSI, CCI, MACD de la **derniere bougie 30min fermee**.
Ces valeurs restent identiques pour les 6 bougies 5min qui suivent (forward-fill).

### Step Index (1 colonne)
Position de la bougie 5min dans la tranche 30min.

**Calcul:**
```python
step = ((minutes % 30) // 5) + 1  # Resultat: 1, 2, 3, 4, 5, 6
```

**Normalisation:**
| Step | Minutes | Valeur normalisee |
|------|---------|-------------------|
| 1 | :00 | 0.17 |
| 2 | :05 | 0.33 |
| 3 | :10 | 0.50 |
| 4 | :15 | 0.67 |
| 5 | :20 | 0.83 |
| 6 | :25 | 1.00 |

---

## 2. Cible Y (Labels)

La "Verite Terrain" reste ancree sur la resolution superieure pour filtrer le bruit.

- **Nature**: Pente du filtre Kalman Non-Causal applique sur les indicateurs 30min
- **Formule**: `Y = 1 si Slope_30min[t] > 0, sinon 0`
- **Alignement**: Pour un bloc [10:00 - 10:30], les 6 bougies 5min partagent le meme label (la pente finale de 10:30)

---

## 3. Workflow de Preparation des Donnees

```
1. Charger donnees 5min
   |
2. Resample 5min → 30min
   |
3. Calculer indicateurs
   ├── 5min: RSI, CCI, MACD (3 colonnes)
   └── 30min: RSI, CCI, MACD (3 colonnes)
   |
4. Synchronisation temporelle
   ├── Forward-fill indicateurs 30min sur timestamps 5min
   └── Generer Step_Index base sur timestamp
   |
5. Generer labels (pente Kalman sur indicateurs 30min)
   |
6. Creer sequences (12 timesteps)
   |
7. Split chronologique avec GAP (12 sequences d'ecart)
   |
8. Sauvegarder .npz
```

---

## 4. Schema de l'Input

```
Timestamp   | 5min Features      | 30min Fixed        | Step |
            | RSI  CCI  MACD     | RSI  CCI  MACD     | Idx  |
------------|--------------------|--------------------|------|
10:00       | 65   52   0.3      | 60   48   0.2      | 0.17 |
10:05       | 67   55   0.4      | 60   48   0.2      | 0.33 |
10:10       | 70   58   0.5      | 60   48   0.2      | 0.50 |
10:15       | 68   56   0.4      | 60   48   0.2      | 0.67 |
10:20       | 66   53   0.3      | 60   48   0.2      | 0.83 |
10:25       | 65   50   0.2      | 60   48   0.2      | 1.00 |
------------|--------------------|--------------------|------|
10:30       | 64   49   0.1      | 65   50   0.2      | 0.17 | ← Nouvelle bougie 30min
...
```

---

## 5. Avantages

1. **Zero Retard Percu**: Le Step Index indique au modele "de combien" les indicateurs 30min sont en retard. Le CNN-LSTM apprend a compenser automatiquement.

2. **Performance de Calcul**: Pas de recalcul Kalman recursif complexe. Le dataset se genere en quelques secondes.

3. **Stabilite**:
   - 5min = reactivite (detection des departs de mouvement)
   - 30min = stabilite (confirmation de tendance)

4. **Horloge Interne**: A step 6, le modele sait que la bougie 30min est presque complete → prediction plus confiante.

---

## 6. Architecture du Modele

```
Input: (batch, 12, 7)
       ├── 5min (3): RSI, CCI, MACD
       ├── 30min (3): RSI, CCI, MACD (forward-filled)
       └── Step Index (1): 0.17 → 1.00
  |
CNN 1D (64 filters)
  |
LSTM (64 hidden x 2)
  |
Dense partage (32)
  |
3 tetes independantes
  |
Output: (batch, 3)  ← RSI, CCI, MACD slopes
```

---

## 7. Objectif de Performance

| Metrique | Actuel | Cible |
|----------|--------|-------|
| Accuracy moyenne | ~76% | **85-90%** |
| Gap train/val | <10% | <10% |

L'injection de l'horloge interne (Step Index) devrait permettre au modele de mieux anticiper les changements de tendance.

---

## 8. Commande d'Execution (a implementer)

```bash
# Preparation des donnees
python src/prepare_data_clock.py --filter kalman --assets BTC ETH BNB ADA LTC

# Entrainement
python src/train.py --data data/prepared/dataset_clock_injected_kalman.npz --epochs 50
```

---

## 9. Differences avec l'Approche Precedente

| Aspect | Ancienne (30min forward-fill) | Clock-Injected |
|--------|-------------------------------|----------------|
| Features 5min | Oui | Oui |
| Features 30min | Forward-fill simple | Forward-fill + Step Index |
| Horloge | Non | Oui (Step 1-6) |
| Compensation lag | Non | Automatique via Step |
| Complexite | Simple | Moderee |

---

**Cree par**: Claude Code
**Derniere MAJ**: 2026-01-02
