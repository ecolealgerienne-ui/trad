"""
Test du module indicators avec des données synthétiques.
"""

import numpy as np
import pandas as pd
import logging
import sys
sys.path.insert(0, 'src')

from indicators import (
    calculate_all_indicators_for_model,
    generate_all_labels,
    create_sequences,
    prepare_datasets
)

# Configurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("TEST MODULE INDICATORS (Données Synthétiques)")
logger.info("="*80)

# Générer données synthétiques
n_samples = 5000
np.random.seed(42)

# Créer un DataFrame avec OHLC
logger.info(f"\n1. Génération de {n_samples:,} bougies synthétiques...")

base_price = 40000
prices = base_price + np.cumsum(np.random.randn(n_samples) * 100)

df = pd.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='5min'),
    'open': prices + np.random.randn(n_samples) * 10,
    'high': prices + np.abs(np.random.randn(n_samples) * 20),
    'low': prices - np.abs(np.random.randn(n_samples) * 20),
    'close': prices,
    'volume': np.random.randint(100, 1000, n_samples)
})

logger.info(f"✅ DataFrame créé: {df.shape}")
logger.info(f"   Colonnes: {list(df.columns)}")
logger.info(f"   Prix range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")

# Split temporel simple
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)

train_df = df[:train_size].copy()
val_df = df[train_size:train_size+val_size].copy()
test_df = df[train_size+val_size:].copy()

logger.info(f"\n2. Split temporel:")
logger.info(f"   Train: {len(train_df):,} bougies")
logger.info(f"   Val:   {len(val_df):,} bougies")
logger.info(f"   Test:  {len(test_df):,} bougies")

# Test du pipeline
logger.info(f"\n3. Test du pipeline complet...")
datasets = prepare_datasets(train_df, val_df, test_df)

# Récupérer datasets
X_train, Y_train = datasets['train']
X_val, Y_val = datasets['val']
X_test, Y_test = datasets['test']

# Afficher résultats
logger.info("\n" + "="*80)
logger.info("RÉSULTATS")
logger.info("="*80)

logger.info(f"\n📊 SHAPES:")
logger.info(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
logger.info(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
logger.info(f"  Test:  X={X_test.shape}, Y={X_test.shape}")

# Validation
logger.info(f"\n🔍 VALIDATION:")
logger.info(f"  X range: [{X_train.min():.2f}, {X_train.max():.2f}]")
logger.info(f"  Y unique values: {np.unique(Y_train)}")

# Vérifier que X est entre 0 et 100
assert X_train.min() >= 0 and X_train.max() <= 100, "X devrait être entre 0 et 100"
logger.info(f"  ✅ X bien normalisé (0-100)")

# Vérifier que Y est binaire
assert set(np.unique(Y_train)) == {0, 1}, "Y devrait contenir seulement 0 et 1"
logger.info(f"  ✅ Y bien binaire (0/1)")

# Vérifier les shapes
assert X_train.shape[1] == 12, "Sequence length devrait être 12"
assert X_train.shape[2] == 4, "Devrait avoir 4 indicateurs"
assert Y_train.shape[1] == 4, "Devrait avoir 4 outputs"
logger.info(f"  ✅ Shapes correctes (12 timesteps, 4 indicateurs, 4 outputs)")

# Stats labels
logger.info(f"\n📈 DISTRIBUTION LABELS (Train):")
for i, name in enumerate(['RSI', 'CCI', 'BOL', 'MACD']):
    buy_count = np.sum(Y_train[:, i])
    buy_pct = buy_count / len(Y_train) * 100
    logger.info(f"  {name}: {buy_count:,} BUY ({buy_pct:.1f}%), "
               f"{len(Y_train) - buy_count:,} SELL ({100-buy_pct:.1f}%)")

logger.info("\n" + "="*80)
logger.info("✅ TOUS LES TESTS PASSENT!")
logger.info("✅ Module indicators.py opérationnel")
logger.info("✅ Prêt pour l'entraînement du modèle CNN-LSTM")
logger.info("="*80)
