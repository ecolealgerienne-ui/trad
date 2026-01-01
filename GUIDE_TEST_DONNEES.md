# Guide : Tester le Chargement des Données

## 📂 Emplacement des Données

Tes données doivent être dans le dossier `../data_trad/` (relatif au projet) :

```
/home/amar/
├── projects/
│   └── trad/          ← Projet
└── data_trad/         ← Données (au même niveau que projects/)
    ├── BTCUSD_all_5m.csv
    └── ETHUSD_all_5m.csv
```

## ✅ Format des Données (Validé)

Le code gère maintenant le format réel :

```csv
Date;Open;High;Low;Close
1577836800000;7170.14;7180.50;7165.00;7175.20
1577837100000;7175.20;7190.00;7172.00;7185.50
...
```

**Caractéristiques** :
- Séparateur : `;` (point-virgule)
- Colonnes : `Date`, `Open`, `High`, `Low`, `Close` (majuscules)
- Timestamp : epoch millisecondes (ex: `1577836800000`)
- Pas de colonne `Volume` (ajoutée automatiquement)

## 🧪 Test Rapide

### 1. Vérifier l'emplacement des fichiers

```bash
cd ~/projects/trad
ls -lh ../data_trad/
```

**Attendu** :
```
-rw-r--r-- 1 amar amar 8.8M Jul 11  2021 BTCUSD_all_5m.csv
-rw-r--r-- 1 amar amar 8.1M Jul 15  2021 ETHUSD_all_5m.csv
```

### 2. Tester le chargement

```bash
python src/data_utils.py
```

**Sortie attendue** :
```
================================================================================
CHARGEMENT ET PRÉPARATION DES DONNÉES
================================================================================
📂 Chargement BTC : ../data_trad/BTCUSD_all_5m.csv
  ⚠️ Colonne 'volume' absente, ajoutée avec valeur par défaut
  → 100,000 dernières bougies chargées
✂️ Trim edges : 100,000 → 99,800 bougies
  Enlevé : 100 début + 100 fin
📂 Chargement ETH : ../data_trad/ETHUSD_all_5m.csv
  ⚠️ Colonne 'volume' absente, ajoutée avec valeur par défaut
  → 100,000 dernières bougies chargées
✂️ Trim edges : 100,000 → 99,800 bougies
  Enlevé : 100 début + 100 fin
🔗 Combinaison BTC + ETH : 199,600 bougies totales
📊 Split temporel (SANS shuffle global - évite data leakage):
  Train: 139,720 bougies (70%) - indices [0:139720]
  Val:   29,940 bougies (15%) - indices [139720:169660]
  Test:  29,940 bougies (15%) - indices [169660:199600]
  ✅ Train shuffled (mélange batches, pas de leakage)
🔍 Validation : Vérification data leakage...
  ✅ Pas de data leakage : timestamps bien séparés
    Train max: 2021-XX-XX XX:XX:XX
    Val range: 2021-XX-XX XX:XX:XX → 2021-XX-XX XX:XX:XX
    Test min: 2021-XX-XX XX:XX:XX
✅ Validation réussie : données propres
================================================================================
✅ DONNÉES PRÊTES
================================================================================

📊 STATS FINALES:
  Train: 139,720 bougies
  Val:   29,940 bougies
  Test:  29,940 bougies
  Total: 199,600 bougies
```

### 3. Vérifier les stats

Le script affiche :
- ✅ Nombre de bougies chargées
- ✅ Période temporelle (première → dernière bougie)
- ✅ Split temporel correct (pas de data leakage)
- ✅ Validation timestamps

## 🔧 Dépannage

### Erreur : `FileNotFoundError`

```
FileNotFoundError: Fichier non trouvé : ../data_trad/BTCUSD_all_5m.csv
```

**Solution** : Vérifier le chemin

```bash
# Depuis le dossier du projet
pwd  # Devrait être /home/amar/projects/trad

# Vérifier que data_trad est au bon endroit
ls ../data_trad/

# Si les fichiers sont ailleurs, créer un symlink
ln -s /chemin/vers/tes/donnees ../data_trad
```

### Erreur : `ValueError: Colonnes manquantes`

```
ValueError: Colonnes manquantes : {'timestamp', 'close', ...}
```

**Solution** : Le format CSV n'est pas reconnu

Vérifier le format :
```bash
head -5 ../data_trad/BTCUSD_all_5m.csv
```

Devrait ressembler à :
```
Date;Open;High;Low;Close
1577836800000;7170.14;...
```

Si format différent, ajuster dans `src/data_utils.py` :
- Ligne 46 : Modifier le séparateur
- Lignes 54-58 : Ajuster `column_mapping`

### Avertissement : `⚠️ Colonne 'volume' absente`

C'est **NORMAL** ! Le volume n'est pas utilisé pour l'instant.

Une colonne `volume` avec valeur par défaut (1.0) est ajoutée automatiquement.

## 📊 Nombre de Bougies Disponibles

Pour savoir combien de bougies tu as :

```bash
# BTC
wc -l ../data_trad/BTCUSD_all_5m.csv
# Résultat : ~XXXXX lignes

# ETH
wc -l ../data_trad/ETHUSD_all_5m.csv
# Résultat : ~XXXXX lignes
```

**Note** : La première ligne est le header, donc nombre de bougies = lignes - 1

## 🎯 Ajuster le Nombre de Bougies

Par défaut, le code charge **100k bougies** de chaque actif.

Pour ajuster, éditer `src/constants.py` :

```python
# Constantes
BTC_CANDLES = 100000  # ← Ajuster ici
ETH_CANDLES = 100000  # ← Ajuster ici
```

Ou passer en paramètre :

```python
from data_utils import load_and_split_btc_eth

# Charger seulement 50k bougies
train, val, test = load_and_split_btc_eth(
    btc_candles=50000,
    eth_candles=50000
)
```

## ✅ Checklist Avant Entraînement

Avant de lancer l'entraînement du modèle, vérifier :

- [ ] Fichiers présents : `../data_trad/BTCUSD_all_5m.csv` et `ETHUSD_all_5m.csv`
- [ ] Format CSV correct (`;` séparateur, colonnes majuscules)
- [ ] Script `python src/data_utils.py` s'exécute sans erreur
- [ ] Pas de data leakage (validation réussie)
- [ ] Nombre de bougies suffisant (minimum ~10k par actif)
- [ ] Split temporel correct (Train → Val → Test chronologique)

## 📚 Prochaines Étapes

Une fois le chargement des données validé :

1. ✅ Calculer les indicateurs (RSI, CCI, BOL, MACD)
2. ✅ Appliquer Decycler parfait (pour labels)
3. ✅ Créer séquences de 12 timesteps
4. ✅ Générer labels (pente 0/1)
5. ✅ Entraîner le modèle CNN-LSTM

---

**Date** : 2026-01-01
**Version** : 1.0
**Status** : Testé et validé avec format réel
