# Approche IA : Prédiction de la Pente des Filtres

## ⚠️ IMPORTANT : Clarification de l'Approche

### Ce qu'on NE fait PAS
❌ **On n'utilise PAS les filtres non-causaux en production**
❌ **On ne prédit PAS les valeurs exactes des filtres**
❌ **On ne cherche PAS à reproduire smooth/filtfilt**

### Ce qu'on FAIT
✅ **On prédit la PENTE (direction) du filtre entre t-1 et t-2**
✅ **Classification binaire : 1 si filtre[t-1] > filtre[t-2], sinon 0**
✅ **L'IA apprend à détecter les changements de direction**

---

## Objectif de l'IA (CNN-LSTM)

### Entrée (Features X)
- Ghost Candles (bougies fantômes 30min échantillonnées à 5min)
- OHLCV 5min
- Indicateurs techniques (RSI, Volume, etc.)
- **Toutes les features sont CAUSALES** (pas de futur)

### Sortie (Label Y)
```python
# Label binaire : Direction de la pente du filtre
Y[t] = 1  si  filter[t-1] > filter[t-2]  # Pente haussière → BUY
Y[t] = 0  si  filter[t-1] <= filter[t-2] # Pente baissière → SELL
```

**Type de problème**: Classification binaire
**Activation finale**: Sigmoid
**Loss function**: Binary Cross-Entropy

---

## Pipeline Complet

### Phase 1 : Génération des Labels (Offline)

```python
# 1. Calculer le filtre de référence (NON-CAUSAL pour créer les labels)
#    On utilise Kalman smooth ou autre filtre parfait pour créer les VRAIS labels
filtered_reference = kalman_filter(close, smooth=True)

# 2. Générer les labels de pente
labels = np.zeros(len(filtered_reference))
for t in range(2, len(filtered_reference)):
    if filtered_reference[t-1] > filtered_reference[t-2]:
        labels[t] = 1  # Pente haussière
    else:
        labels[t] = 0  # Pente baissière

# 3. Ces labels servent à ENTRAÎNER le modèle
```

### Phase 2 : Entraînement du Modèle

```python
# Le modèle apprend à prédire la pente à partir des features causales
model = CNN_LSTM()

# Features : Ghost Candles + OHLCV (CAUSALES)
X_train = ghost_candles[:-trim]

# Labels : Pente du filtre parfait (générés offline)
Y_train = labels[:-trim]

model.fit(X_train, Y_train)
```

### Phase 3 : Utilisation en Production (Online)

```python
# En temps réel, on prédit la pente directement
prediction = model.predict(ghost_candles_current)

# prediction = probabilité que filter[t-1] > filter[t-2]
if prediction > 0.5:
    signal = 'BUY'   # On prédit pente haussière
    position = 1
else:
    signal = 'SELL'  # On prédit pente baissière
    position = -1

# Trade à open[t+1]
trade_price = open[t+1]
```

---

## Pourquoi cette Approche ?

### Avantages

1. **Simplicité**
   - Classification binaire (plus simple que régression)
   - Pas besoin de prédire valeurs exactes du filtre
   - Juste prédire la direction (hausse/baisse)

2. **Robustesse**
   - Moins sensible aux outliers
   - Classification est plus stable que régression
   - Métriques claires (Accuracy, Precision, Recall)

3. **Réalisme**
   - On prédit seulement ce dont on a besoin (direction)
   - Pas de sur-engineering
   - Compatible avec trading réel

4. **Performance**
   - Modèle plus léger (classification vs régression)
   - Inference plus rapide
   - Moins de paramètres à optimiser

### Lien avec les Tests "Monde Parfait"

Les tests avec filtres non-causaux (`test_perfect_world.py`) servent à :

✅ **VALIDER** que la méthode `filter[t-1] > filter[t-2]` fonctionne
✅ **PROUVER** qu'avec un filtre exact, on obtient Profit Factor > 7.44
✅ **JUSTIFIER** pourquoi on investit dans l'IA pour prédire cette pente

**Mais en production**, on n'utilise PAS les filtres non-causaux. On utilise l'IA qui prédit la pente.

---

## Métriques de Succès de l'IA

### Pendant l'Entraînement

```python
# Classification metrics
accuracy = (predictions == labels).mean()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
```

**Target** : Accuracy > 55-60% (au-dessus du hasard 50%)

### En Backtesting

```python
# Trading metrics (avec les prédictions de l'IA)
profit_factor = gross_profit / gross_loss
win_rate = winning_trades / total_trades
sharpe_ratio = mean_return / std_return * sqrt(252)
```

**Target** : Profit Factor > 1.5-2.0 (réaliste avec IA)

⚠️ **On ne vise PAS 7.44** (monde parfait) mais quelque chose de réaliste !

---

## Architecture du Modèle (Exemple)

```python
class PentePredictorCNN_LSTM(nn.Module):
    """
    Prédit la pente du filtre : 1 si hausse, 0 si baisse.
    """
    def __init__(self):
        super().__init__()

        # CNN pour extraire features locales
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)

        # LSTM pour capturer dynamique temporelle
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2)

        # Classification binaire
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, sequence, features)
        x = x.transpose(1, 2)  # (batch, features, sequence)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.transpose(1, 2)  # (batch, sequence, features)

        out, _ = self.lstm(x)

        # Prendre dernière sortie
        out = out[:, -1, :]

        # Classification
        out = self.fc(out)
        out = self.sigmoid(out)

        return out  # Probabilité de pente haussière
```

**Loss function** :
```python
criterion = nn.BCELoss()  # Binary Cross-Entropy
```

---

## Différence Clef : Monde Parfait vs Production

| Aspect | Monde Parfait (Validation) | Production (IA) |
|--------|---------------------------|-----------------|
| **Objectif** | Valider la méthode théoriquement | Trading réel |
| **Filtre** | Non-causal (smooth/filtfilt) | Pas de filtre ! IA directe |
| **Prédiction** | Valeurs exactes du filtre | Pente (direction) 0/1 |
| **Profit Factor** | 7.44 - 995 (théorique) | 1.5 - 3.0 (réaliste) |
| **Utilisation** | Proof of concept | Système de trading |

---

## Workflow Complet

### 1. Préparation des Données
```python
# Générer Ghost Candles (features causales)
ghost_candles = create_ghost_candles(df_5m, target_timeframe='30min')

# Générer labels (avec filtre parfait OFFLINE)
filtered = kalman_filter(df_5m['close'], smooth=True)  # Non-causal OK ici
labels = (filtered[1:] > filtered[:-1]).astype(int)

# Split train/val/test (avec trim des edges !)
X_train, Y_train = trim_and_split(ghost_candles, labels, trim=100)
```

### 2. Entraînement
```python
model = PentePredictorCNN_LSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(epochs):
    predictions = model(X_train)
    loss = criterion(predictions, Y_train)

    loss.backward()
    optimizer.step()

    # Métriques
    accuracy = ((predictions > 0.5) == Y_train).float().mean()
    print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.2%}")
```

### 3. Backtesting avec IA
```python
# En backtest, utiliser les PRÉDICTIONS de l'IA
predictions = model.predict(X_test)

signals = []
for t in range(2, len(predictions)):
    if predictions[t] > 0.5:
        signal = 'BUY'
    else:
        signal = 'SELL'
    signals.append(signal)

# Tester la stratégie avec ces signaux
results = backtest_strategy(df_test, signals)
print(f"Profit Factor: {results['profit_factor']:.2f}")
```

---

## Résumé

🎯 **Objectif IA** : Prédire si `filter[t-1] > filter[t-2]` (classification binaire)

✅ **Entrée** : Ghost Candles + features causales
✅ **Sortie** : 0 ou 1 (pente baisse ou hausse)
✅ **Entraînement** : Labels générés avec filtre parfait (offline)
✅ **Production** : IA directe, pas de filtre

❌ **On n'utilise PAS** les filtres non-causaux en production
❌ **On ne prédit PAS** les valeurs du filtre
❌ **On ne vise PAS** Profit Factor 7.44 (irréaliste avec IA)

📚 **Référence** : Les tests "monde parfait" valident la méthode, mais le système réel utilise l'IA pour prédire la pente.

---

**Date** : 2026-01-01
**Version** : 1.0
**Status** : Approche validée et documentée
