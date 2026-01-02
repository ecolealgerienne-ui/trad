# 🚀 Guide d'Utilisation - Entraînement CNN-LSTM

## Arguments en Ligne de Commande

Le script `train.py` accepte maintenant des arguments CLI pour personnaliser l'entraînement sans modifier `constants.py`.

### Utilisation de Base

```bash
# Entraînement avec paramètres par défaut
python src/train.py

# Afficher l'aide
python src/train.py --help
```

---

## 📊 Hyperparamètres

### Batch Size

```bash
# Batch size par défaut (32)
python src/train.py

# Batch size personnalisé
python src/train.py --batch-size 64
python src/train.py --batch-size 16  # Pour GPU avec moins de mémoire
```

**Recommandations** :
- **CPU** : 16-32
- **GPU (4GB)** : 32-64
- **GPU (8GB+)** : 64-128

### Learning Rate

```bash
# Learning rate par défaut (0.001)
python src/train.py

# Learning rate personnalisé
python src/train.py --lr 0.0001  # Plus conservateur
python src/train.py --lr 0.01    # Plus agressif
```

**Recommandations** :
- Commencer avec **0.001**
- Si loss oscille : réduire à **0.0001**
- Si convergence trop lente : augmenter à **0.005**

### Nombre d'Époques

```bash
# Nombre d'époques par défaut (100)
python src/train.py

# Nombre d'époques personnalisé
python src/train.py --epochs 50   # Entraînement court
python src/train.py --epochs 200  # Entraînement long
```

### Early Stopping Patience

```bash
# Patience par défaut (10)
python src/train.py

# Patience personnalisée
python src/train.py --patience 5   # Arrête plus rapidement
python src/train.py --patience 20  # Plus de tolérance
```

---

## 🔬 Type de Filtre pour Labels

```bash
# Filtre par défaut (Decycler)
python src/train.py

# Forcer filtre Decycler
python src/train.py --filter decycler

# Utiliser filtre Kalman (meilleure qualité)
python src/train.py --filter kalman
```

**Comparaison des filtres** :
- **Decycler** : Filtre de Ehlers, rapide, ~67% accuracy
- **Kalman** : Kalman smoothing, meilleure qualité, ~85% accuracy

**Recommandation** : Utiliser `--filter kalman` pour de meilleurs résultats.

---

## 💻 Device (CPU/GPU)

```bash
# Auto-détection (par défaut)
python src/train.py

# Forcer CPU
python src/train.py --device cpu

# Forcer GPU
python src/train.py --device cuda
```

---

## 💾 Chemins de Sauvegarde

```bash
# Chemin par défaut (models/best_model.pth)
python src/train.py

# Chemin personnalisé
python src/train.py --save-path models/experiment_1.pth
python src/train.py --save-path models/cnn_lstm_v2.pth
```

---

## 🎲 Random Seed

```bash
# Seed par défaut (42)
python src/train.py

# Seed personnalisé (pour reproductibilité)
python src/train.py --seed 123
```

---

## 🔥 Exemples Pratiques

### Test Rapide (CPU, petit batch)

```bash
python src/train.py \
    --batch-size 16 \
    --epochs 10 \
    --device cpu
```

### Entraînement Standard (GPU)

```bash
python src/train.py \
    --batch-size 64 \
    --lr 0.001 \
    --epochs 100 \
    --patience 10 \
    --filter kalman
```

### Entraînement Long (GPU puissant)

```bash
python src/train.py \
    --batch-size 128 \
    --lr 0.001 \
    --epochs 200 \
    --patience 15 \
    --filter kalman
```

### Fine-tuning avec Learning Rate Bas

```bash
python src/train.py \
    --batch-size 32 \
    --lr 0.0001 \
    --epochs 50 \
    --patience 20
```

### Expérimentation (sauvegarder dans un fichier différent)

```bash
python src/train.py \
    --batch-size 64 \
    --lr 0.005 \
    --epochs 100 \
    --save-path models/experiment_lr005.pth \
    --seed 999
```

---

## 📈 Monitoring Pendant l'Entraînement

### GPU

Dans un terminal séparé :

```bash
watch -n 2 nvidia-smi
```

### Logs

Le script affiche en temps réel :
- Train Loss / Accuracy / F1
- Val Loss / Accuracy / F1
- Meilleur modèle sauvegardé

---

## ⚠️ Troubleshooting

### Out of Memory (GPU)

```bash
# Réduire batch size
python src/train.py --batch-size 16

# Ou forcer CPU
python src/train.py --device cpu
```

### Convergence Lente

```bash
# Augmenter learning rate
python src/train.py --lr 0.005

# Ou augmenter nombre d'époques
python src/train.py --epochs 200
```

### Loss Oscille

```bash
# Réduire learning rate
python src/train.py --lr 0.0001

# Augmenter batch size (plus stable)
python src/train.py --batch-size 128
```

### Overfitting (Val Loss monte)

```bash
# Réduire patience (arrête plus tôt)
python src/train.py --patience 5

# Ou augmenter données (modifier constants.py)
```

---

## 📚 Valeurs par Défaut

Définies dans `src/constants.py` :

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42
```

---

## ✅ Vérification Post-Entraînement

```bash
# Évaluer le modèle
python src/evaluate.py

# Vérifier les fichiers générés
ls -lh models/
ls -lh results/

# Visualiser l'historique
cat models/training_history.json
```

---

**Créé le** : 2026-01-01
**Version** : 1.0
