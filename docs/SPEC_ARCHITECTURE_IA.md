# Spécification Architecture IA - Multi-Output Predictor

## 📋 Vue d'Ensemble

**Objectif** : Prédire la pente (direction) de 4 indicateurs techniques filtrés

**Type** : Classification multi-output (4 sorties binaires)

**Input** : 4 indicateurs × 12 timesteps (séquence temporelle)

**Output** : 4 probabilités (une par indicateur)

---

## 🎯 Réponses aux Questions Techniques

### 1. Normalisation du CCI

**Plage brute du CCI** : ~-200 à +200 (typiquement)

**Méthode de normalisation** : Min-Max Scaling

```python
# Constantes
CCI_RAW_MIN = -200
CCI_RAW_MAX = 200
INDICATOR_MIN = 0
INDICATOR_MAX = 100

# Normalisation
CCI_normalized = ((CCI - CCI_RAW_MIN) / (CCI_RAW_MAX - CCI_RAW_MIN)) * 100
```

**Exemple** :
- CCI = -200 → CCI_norm = 0
- CCI = 0 → CCI_norm = 50
- CCI = +200 → CCI_norm = 100

**Gestion des outliers** :
```python
# Clipper les valeurs extrêmes
CCI_clipped = np.clip(CCI, CCI_RAW_MIN, CCI_RAW_MAX)
CCI_normalized = ((CCI_clipped - CCI_RAW_MIN) / (CCI_RAW_MAX - CCI_RAW_MIN)) * 100
```

---

### 2. Bollinger %B

**Indicateur utilisé** : **%B (Percent B)**

```python
# %B = Position du prix dans les bandes
%B = (close - lower_band) / (upper_band - lower_band)
```

**Plage** :
- %B = 0 : Prix sur la bande inférieure
- %B = 0.5 : Prix sur la moyenne mobile (centre)
- %B = 1.0 : Prix sur la bande supérieure

**Normalisation** :
```python
# %B déjà entre 0 et 1, on multiplie par 100
BOL_normalized = %B * 100
# Clipper pour gérer les outliers (prix hors bandes)
BOL_normalized = np.clip(BOL_normalized, 0, 100)
```

**Avantages de %B** :
- ✅ Normalisé automatiquement (0-1)
- ✅ Capture volatilité + position du prix
- ✅ Indépendant du prix absolu

---

### 3. MACD

**Indicateur utilisé** : **MACD Histogram**

```python
# MACD line
MACD_line = EMA(close, MACD_FAST) - EMA(close, MACD_SLOW)

# Signal line
Signal_line = EMA(MACD_line, MACD_SIGNAL)

# Histogram (ce qu'on utilise)
MACD_histogram = MACD_line - Signal_line
```

**Pourquoi histogram ?**
- ✅ Détecte les divergences (changements de momentum)
- ✅ Plus sensible que la MACD line seule
- ✅ Signaux plus clairs

**Normalisation** :
```python
# MACD histogram n'a pas de bornes fixes → normalisation dynamique

# Option A : Min-Max sur fenêtre glissante (RECOMMANDÉ)
window = 1000  # Dernières 1000 bougies
MACD_min = MACD_histogram[-window:].min()
MACD_max = MACD_histogram[-window:].max()
MACD_normalized = ((MACD_histogram - MACD_min) / (MACD_max - MACD_min)) * 100

# Option B : Z-score puis clip
MACD_mean = MACD_histogram[-window:].mean()
MACD_std = MACD_histogram[-window:].std()
MACD_zscore = (MACD_histogram - MACD_mean) / MACD_std
MACD_normalized = np.clip((MACD_zscore + 3) / 6 * 100, 0, 100)  # ±3σ → 0-100
```

**Recommandation** : Option A (Min-Max sur fenêtre)

---

### 4. Vote Majoritaire en Production

**Méthode recommandée** : **Moyenne pondérée**

```python
# Prédictions du modèle (4 probabilités entre 0 et 1)
pred_RSI = 0.85   # 85% confiance hausse
pred_CCI = 0.65   # 65% confiance hausse
pred_BOL = 0.72   # 72% confiance hausse
pred_MACD = 0.45  # 45% confiance hausse (baisse probable)

# Option A : Moyenne simple (RECOMMANDÉ pour démarrage)
decision = (pred_RSI + pred_CCI + pred_BOL + pred_MACD) / 4
# decision = 0.6675 > 0.5 → BUY

# Option B : Moyenne pondérée (basée sur performance monde parfait)
weights = {
    'RSI': 0.35,    # PF 41.47 → poids élevé
    'CCI': 0.25,    # PF 11.82
    'BOL': 0.25,    # PF 17.28
    'MACD': 0.15    # PF 3.79 → poids faible
}
decision = (pred_RSI * 0.35 + pred_CCI * 0.25 +
            pred_BOL * 0.25 + pred_MACD * 0.15)

# Option C : Vote strict (au moins 3/4 doivent dire hausse)
count_hausse = sum([pred_RSI > 0.5, pred_CCI > 0.5,
                     pred_BOL > 0.5, pred_MACD > 0.5])
decision = 1 if count_hausse >= 3 else 0
```

**Recommandation** : Commencer avec **Option A** (moyenne simple), puis tester **Option B** si besoin d'optimisation.

---

### 5. Loss Function

**Loss** : Binary Cross-Entropy par sortie, puis moyenne

```python
import torch.nn as nn

# Loss function
criterion = nn.BCELoss()

# Forward pass
outputs = model(X)  # [pred_RSI, pred_CCI, pred_BOL, pred_MACD]
labels = [Y_RSI, Y_CCI, Y_BOL, Y_MACD]

# Calculer loss pour chaque sortie
loss_RSI = criterion(outputs[0], labels[0])
loss_CCI = criterion(outputs[1], labels[1])
loss_BOL = criterion(outputs[2], labels[2])
loss_MACD = criterion(outputs[3], labels[3])

# Loss totale (moyenne simple ou pondérée)
# Option A : Moyenne simple
loss_total = (loss_RSI + loss_CCI + loss_BOL + loss_MACD) / 4

# Option B : Moyenne pondérée (basée sur importance)
weights = [1.0, 1.0, 1.0, 1.0]  # Égal pour commencer
loss_total = (loss_RSI * weights[0] + loss_CCI * weights[1] +
              loss_BOL * weights[2] + loss_MACD * weights[3]) / sum(weights)
```

**Recommandation** : **Option A** (poids égaux) pour commencer.

---

### 6. Poids des Sorties

**Approche** : Tous les indicateurs ont le **même poids** au départ

**Raison** :
1. Laisser l'IA apprendre l'importance relative
2. Éviter de biaiser le modèle avant entraînement
3. On peut pondérer APRÈS (vote majoritaire) si besoin

**Constantes** :
```python
LOSS_WEIGHT_RSI = 1.0
LOSS_WEIGHT_CCI = 1.0
LOSS_WEIGHT_BOL = 1.0
LOSS_WEIGHT_MACD = 1.0
```

**Si on veut optimiser plus tard** :
```python
# Après validation, on peut ajuster basé sur performance
LOSS_WEIGHT_RSI = 1.5   # Meilleur PF → poids plus élevé
LOSS_WEIGHT_CCI = 1.0
LOSS_WEIGHT_BOL = 1.2
LOSS_WEIGHT_MACD = 0.8  # Plus faible PF → poids réduit
```

---

### 7. Données ETH

**Timeframe** : **5 minutes** (identique à BTC)

**Nombre de bougies** : **~100k** (identique à BTC)

**Synchronisation** : Pas nécessaire de synchroniser BTC et ETH temporellement

**Pourquoi** :
- L'IA apprend des **patterns génériques**, pas des corrélations BTC-ETH
- Les bougies sont mélangées aléatoirement pendant l'entraînement
- Chaque bougie est traitée indépendamment

---

### 8. Augmentation de Données (Mix BTC + ETH)

**Méthode** : **Split temporel STRICT** (pas de shuffle global!)

```python
# Charger BTC et ETH
btc_data = load_btc_data(n=100000)  # 100k bougies
eth_data = load_eth_data(n=100000)  # 100k bougies

# Concaténer
all_data = pd.concat([btc_data, eth_data], ignore_index=True)
# Total : 200k bougies

# ⚠️ CRITIQUE: Split TEMPOREL (PAS de shuffle avant split!)
# Évite le data leakage (séquences proches dans train ET test)
n_total = len(all_data)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)

train_data = all_data[:n_train]           # 70% premiers
val_data = all_data[n_train:n_train+n_val]  # 15% suivants
test_data = all_data[n_train+n_val:]      # 15% derniers

# Shuffle APRÈS split (uniquement dans train pour mélanger batches)
train_data = train_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
```

**Pourquoi split temporel ?**
- ✅ **Évite data leakage** : Pas de séquences t et t+1 dans train ET test
- ✅ **Réaliste** : L'IA s'entraîne sur le passé, valide sur le futur
- ✅ **Simule production** : En prod, on prédit le futur, pas le passé

**Pourquoi PAS shuffle global ?**
- ❌ **Data leakage massif** : Fenêtres de 12 timesteps qui se chevauchent
- ❌ **Triche** : L'IA reconnaît contexte immédiat (vu en train)
- ❌ **Fausse accuracy** : 90%+ en test mais 50% en production

**Shuffle APRÈS** (dans train seulement) :
- ✅ Mélange l'ordre des batches
- ✅ Évite biais d'ordre (BTC puis ETH)
- ✅ N'introduit PAS de leakage (déjà split temporellement)

---

## 🏗️ Architecture Détaillée du Modèle

### Input Shape

```python
# Input
X = torch.tensor([
    [RSI[t-12:t], CCI[t-12:t], BOL[t-12:t], MACD[t-12:t]]
])
# Shape : (batch_size, sequence_length, num_indicators)
#       = (batch_size, 12, 4)
```

### Architecture CNN-LSTM Multi-Output

```python
import torch
import torch.nn as nn

class MultiIndicatorPredictor(nn.Module):
    """
    Prédicteur multi-output pour 4 indicateurs techniques.

    Input : (batch, 12 timesteps, 4 features)
    Output : 4 probabilités (une par indicateur)
    """

    def __init__(self):
        super().__init__()

        # === CONSTANTES ===
        from constants import (
            SEQUENCE_LENGTH, NUM_INDICATORS, NUM_OUTPUTS,
            CNN_FILTERS, CNN_KERNEL_SIZE,
            LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
            DENSE_HIDDEN_SIZE, DENSE_DROPOUT
        )

        # === CNN LAYER (extraction features temporelles) ===
        # Input : (batch, 12, 4) → transposé → (batch, 4, 12)
        self.conv1 = nn.Conv1d(
            in_channels=NUM_INDICATORS,
            out_channels=CNN_FILTERS,
            kernel_size=CNN_KERNEL_SIZE,
            padding=CNN_KERNEL_SIZE // 2  # Same padding
        )
        self.bn1 = nn.BatchNorm1d(CNN_FILTERS)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.2)

        # === LSTM LAYER (capture dynamique temporelle) ===
        # Input : (batch, seq, features) = (batch, 12, 64)
        self.lstm = nn.LSTM(
            input_size=CNN_FILTERS,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            batch_first=True
        )

        # === DENSE SHARED (features communes) ===
        self.dense_shared = nn.Linear(LSTM_HIDDEN_SIZE, DENSE_HIDDEN_SIZE)
        self.bn_dense = nn.BatchNorm1d(DENSE_HIDDEN_SIZE)
        self.dropout_dense = nn.Dropout(DENSE_DROPOUT)

        # === 4 TÊTES DE SORTIE (une par indicateur) ===
        self.head_RSI = nn.Linear(DENSE_HIDDEN_SIZE, 1)
        self.head_CCI = nn.Linear(DENSE_HIDDEN_SIZE, 1)
        self.head_BOL = nn.Linear(DENSE_HIDDEN_SIZE, 1)
        self.head_MACD = nn.Linear(DENSE_HIDDEN_SIZE, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x : (batch, sequence_length, num_indicators)
              = (batch, 12, 4)

        Returns:
            List de 4 tensors (batch, 1) - probabilités par indicateur
        """
        batch_size = x.size(0)

        # === CNN ===
        # Transpose pour Conv1d : (batch, 12, 4) → (batch, 4, 12)
        x = x.transpose(1, 2)

        # Conv + BatchNorm + ReLU + Dropout
        x = self.conv1(x)  # (batch, 64, 12)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_cnn(x)

        # === LSTM ===
        # Transpose back : (batch, 64, 12) → (batch, 12, 64)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out : (batch, 12, 64)
        # h_n : (num_layers, batch, 64)

        # Prendre dernière sortie temporelle
        x = lstm_out[:, -1, :]  # (batch, 64)

        # === DENSE SHARED ===
        x = self.dense_shared(x)  # (batch, 32)
        x = self.bn_dense(x)
        x = self.relu(x)
        x = self.dropout_dense(x)

        # === 4 SORTIES INDÉPENDANTES ===
        out_RSI = self.sigmoid(self.head_RSI(x))    # (batch, 1)
        out_CCI = self.sigmoid(self.head_CCI(x))    # (batch, 1)
        out_BOL = self.sigmoid(self.head_BOL(x))    # (batch, 1)
        out_MACD = self.sigmoid(self.head_MACD(x))  # (batch, 1)

        return [out_RSI, out_CCI, out_BOL, out_MACD]
```

### Training Loop

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Entraînement du modèle multi-output."""

    from constants import LOSS_WEIGHT_RSI, LOSS_WEIGHT_CCI, LOSS_WEIGHT_BOL, LOSS_WEIGHT_MACD

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_Y in train_loader:
            # batch_X : (batch, 12, 4)
            # batch_Y : (batch, 4) - 4 labels binaires

            optimizer.zero_grad()

            # Forward
            outputs = model(batch_X)  # [out_RSI, out_CCI, out_BOL, out_MACD]

            # Calculer loss pour chaque sortie
            loss_RSI = criterion(outputs[0], batch_Y[:, 0:1])  # (batch, 1)
            loss_CCI = criterion(outputs[1], batch_Y[:, 1:2])
            loss_BOL = criterion(outputs[2], batch_Y[:, 2:3])
            loss_MACD = criterion(outputs[3], batch_Y[:, 3:4])

            # Loss totale (moyenne pondérée)
            loss = (loss_RSI * LOSS_WEIGHT_RSI +
                    loss_CCI * LOSS_WEIGHT_CCI +
                    loss_BOL * LOSS_WEIGHT_BOL +
                    loss_MACD * LOSS_WEIGHT_MACD) / 4

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2%}")
```

### Inference (Production)

```python
def predict_and_vote(model, current_data):
    """
    Prédiction et vote majoritaire.

    Args:
        current_data : (12, 4) - derniers 12 timesteps des 4 indicateurs

    Returns:
        decision : 0 (SELL) ou 1 (BUY)
        confidence : float entre 0 et 1
    """
    model.eval()

    with torch.no_grad():
        # Ajouter dimension batch
        x = torch.tensor(current_data).unsqueeze(0).float()  # (1, 12, 4)

        # Prédiction
        outputs = model(x)  # [out_RSI, out_CCI, out_BOL, out_MACD]

        # Extraire probabilités
        pred_RSI = outputs[0].item()
        pred_CCI = outputs[1].item()
        pred_BOL = outputs[2].item()
        pred_MACD = outputs[3].item()

        # Vote majoritaire (moyenne simple)
        confidence = (pred_RSI + pred_CCI + pred_BOL + pred_MACD) / 4
        decision = 1 if confidence > 0.5 else 0

        return decision, confidence, {
            'RSI': pred_RSI,
            'CCI': pred_CCI,
            'BOL': pred_BOL,
            'MACD': pred_MACD
        }

# Exemple d'utilisation
decision, confidence, details = predict_and_vote(model, current_indicators)

if decision == 1:
    print(f"BUY (confiance: {confidence:.2%})")
    print(f"  RSI: {details['RSI']:.2%}")
    print(f"  CCI: {details['CCI']:.2%}")
    print(f"  BOL: {details['BOL']:.2%}")
    print(f"  MACD: {details['MACD']:.2%}")
else:
    print(f"SELL (confiance: {1-confidence:.2%})")
```

---

## 📝 Résumé des Choix Techniques

| Aspect | Choix | Justification |
|--------|-------|---------------|
| **CCI normalisation** | Min-Max (-200/+200 → 0-100) | Simple, robuste, cohérent avec autres indicateurs |
| **Bollinger** | %B × 100 | Déjà normalisé, capture volatilité + position |
| **MACD** | Histogram, Min-Max fenêtre 1000 | Plus sensible, normalisation dynamique |
| **Vote prod** | Moyenne simple (4 sorties) | Simple, efficace, évolutif |
| **Loss** | BCE moyenne (poids égaux) | Standard, pas de biais initial |
| **Poids sorties** | Égaux (1.0 partout) | Laisser IA apprendre, optimiser après |
| **ETH data** | 5min, 100k bougies | Identique à BTC |
| **Mix BTC+ETH** | Shuffle aléatoire | Évite biais temporel, meilleure généralisation |

---

**Date** : 2026-01-01
**Version** : 1.0
**Status** : Spécifications validées
