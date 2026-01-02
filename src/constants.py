"""
Constantes du Projet de Prédiction de Trading

Ce fichier centralise toutes les constantes utilisées dans le projet.
Toute modification de paramètres doit être faite ICI uniquement.
"""

# =============================================================================
# CONSTANTES DONNÉES
# =============================================================================

# Timeframe
TIMEFRAME_MINUTES = 5  # Bougies 5 minutes
CANDLES_PER_HOUR = 60 // TIMEFRAME_MINUTES  # 12 bougies par heure
CANDLES_PER_DAY = CANDLES_PER_HOUR * 24  # 288 bougies par jour

# Dataset
SEQUENCE_LENGTH = 12  # Nombre de timesteps pour l'input (t-12:t)
TRIM_EDGES = 100  # Valeurs à enlever au début + fin (warm-up + artifacts)

# Multi-actifs
BTC_CANDLES = 100000  # Nombre de bougies BTC à charger
ETH_CANDLES = 100000  # Nombre de bougies ETH à charger

# =============================================================================
# CONSTANTES INDICATEURS TECHNIQUES
# =============================================================================

# RSI (Relative Strength Index)
RSI_PERIOD = 5  # Période agressive pour capturer les mouvements rapides

# CCI (Commodity Channel Index)
CCI_PERIOD = 7  # Période agressive pour réactivité
CCI_CONSTANT = 0.015  # Constante de scaling du CCI

# Bollinger Bands
BOL_PERIOD = 20  # Période de la moyenne mobile
BOL_NUM_STD = 2  # Nombre d'écarts-types pour les bandes

# MACD (Moving Average Convergence Divergence)
MACD_FAST = 5  # Période EMA rapide (agressive)
MACD_SLOW = 13  # Période EMA lente (agressive)
MACD_SIGNAL = 9  # Période de la ligne de signal

# =============================================================================
# CONSTANTES NORMALISATION
# =============================================================================

# Plages de normalisation cible (tous les indicateurs → 0-100)
INDICATOR_MIN = 0
INDICATOR_MAX = 100

# Plages brutes des indicateurs (pour normalisation)
RSI_RAW_MIN = 0
RSI_RAW_MAX = 100  # RSI déjà entre 0-100

CCI_RAW_MIN = -200  # CCI typique min
CCI_RAW_MAX = 200  # CCI typique max

BOL_RAW_MIN = 0  # %B entre 0 et 1 (puis × 100)
BOL_RAW_MAX = 1

# MACD : normalisé dynamiquement (min-max sur window)
MACD_NORM_WINDOW = 1000  # Fenêtre pour calculer min/max du MACD

# =============================================================================
# CONSTANTES FILTRES (pour génération des labels)
# =============================================================================

# Decycler (Ehlers)
DECYCLER_CUTOFF = 0.1  # Fréquence de coupure

# Kalman (pour labels monde parfait)
KALMAN_PROCESS_VAR = 0.01  # Variance du processus (Q)
KALMAN_MEASURE_VAR = 0.1  # Variance de mesure (R)

# Choix du filtre pour génération labels
LABEL_FILTER_TYPE = 'decycler'  # 'decycler' ou 'kalman'

# =============================================================================
# CONSTANTES MODÈLE CNN-LSTM
# =============================================================================

# Architecture
NUM_INDICATORS = 4  # RSI, CCI, BOL, MACD
NUM_OUTPUTS = 4  # Une sortie par indicateur (multi-output)

# CNN
CNN_FILTERS = 64  # Nombre de filtres CNN
CNN_KERNEL_SIZE = 3  # Taille du kernel
CNN_STRIDE = 1  # Stride
CNN_PADDING = 1  # Padding (same)

# LSTM
LSTM_HIDDEN_SIZE = 64  # Taille de la couche cachée
LSTM_NUM_LAYERS = 2  # Nombre de couches LSTM
LSTM_DROPOUT = 0.2  # Dropout entre les couches LSTM

# Dense layers
DENSE_HIDDEN_SIZE = 32  # Taille de la couche dense intermédiaire
DENSE_DROPOUT = 0.3  # Dropout après dense

# =============================================================================
# CONSTANTES ENTRAÎNEMENT
# =============================================================================

# Hyperparamètres
BATCH_SIZE = 32  # Taille du batch
LEARNING_RATE = 0.001  # Taux d'apprentissage (Adam)
NUM_EPOCHS = 100  # Nombre d'époques
EARLY_STOPPING_PATIENCE = 10  # Patience pour early stopping

# Split dataset
TRAIN_SPLIT = 0.7  # 70% train
VAL_SPLIT = 0.15  # 15% validation
TEST_SPLIT = 0.15  # 15% test

# Seed pour reproductibilité
RANDOM_SEED = 42

# Loss weights (si on veut pondérer différemment les sorties)
LOSS_WEIGHT_RSI = 1.0  # Poids pour la loss du RSI
LOSS_WEIGHT_CCI = 1.0  # Poids pour la loss du CCI
LOSS_WEIGHT_BOL = 1.0  # Poids pour la loss du BOL
LOSS_WEIGHT_MACD = 1.0  # Poids pour la loss du MACD

# =============================================================================
# CONSTANTES PRODUCTION / INFÉRENCE
# =============================================================================

# Vote majoritaire
VOTE_THRESHOLD = 0.5  # Seuil de décision (moyenne des 4 prédictions)
MIN_CONFIDENCE = 0.6  # Confiance minimale pour trader (optionnel)

# Gestion des positions
MAX_POSITION_SIZE = 1.0  # Taille maximale de position (100%)
STOP_LOSS_PCT = 0.02  # Stop loss à 2% (optionnel)
TAKE_PROFIT_PCT = 0.05  # Take profit à 5% (optionnel)

# =============================================================================
# CONSTANTES BACKTESTING
# =============================================================================

# Frais de trading
TRADING_FEE_PCT = 0.001  # 0.1% par trade (typique crypto)
SLIPPAGE_PCT = 0.0005  # 0.05% slippage

# Métriques
RISK_FREE_RATE = 0.0  # Taux sans risque pour Sharpe (0% en crypto)
TRADING_DAYS_PER_YEAR = 365  # Crypto trade 24/7

# =============================================================================
# CONSTANTES CHEMINS
# =============================================================================

# Données
DATA_DIR = '../data_trad'  # Dossier des données réelles
RAW_DATA_DIR = '../data_trad'
PROCESSED_DATA_DIR = 'data/processed'

BTC_DATA_FILE = '../data_trad/BTCUSD_all_5m.csv'
ETH_DATA_FILE = '../data_trad/ETHUSD_all_5m.csv'

# Modèles
MODELS_DIR = 'models'
CHECKPOINTS_DIR = 'models/checkpoints'
BEST_MODEL_PATH = 'models/best_model.pth'

# Logs
LOGS_DIR = 'logs'
TENSORBOARD_DIR = 'logs/tensorboard'

# Résultats
RESULTS_DIR = 'results'
PREDICTIONS_DIR = 'results/predictions'
BACKTESTS_DIR = 'results/backtests'

# =============================================================================
# CONSTANTES AFFICHAGE / LOGGING
# =============================================================================

# Verbosity
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
PRINT_EVERY = 100  # Afficher métriques tous les N batches

# Visualisation
PLOT_DPI = 150  # Résolution des graphiques
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# =============================================================================
# VALIDATION DES CONSTANTES
# =============================================================================

def validate_constants():
    """Valide que les constantes sont cohérentes."""
    assert SEQUENCE_LENGTH > 0, "SEQUENCE_LENGTH doit être > 0"
    assert TRIM_EDGES >= 0, "TRIM_EDGES doit être >= 0"

    assert RSI_PERIOD > 0, "RSI_PERIOD doit être > 0"
    assert CCI_PERIOD > 0, "CCI_PERIOD doit être > 0"
    assert BOL_PERIOD > 0, "BOL_PERIOD doit être > 0"
    assert MACD_FAST < MACD_SLOW, "MACD_FAST doit être < MACD_SLOW"

    assert NUM_INDICATORS == 4, "NUM_INDICATORS doit être 4 (RSI, CCI, BOL, MACD)"
    assert NUM_OUTPUTS == NUM_INDICATORS, "NUM_OUTPUTS doit égaler NUM_INDICATORS"

    assert 0 < TRAIN_SPLIT < 1, "TRAIN_SPLIT doit être entre 0 et 1"
    assert 0 < VAL_SPLIT < 1, "VAL_SPLIT doit être entre 0 et 1"
    assert 0 < TEST_SPLIT < 1, "TEST_SPLIT doit être entre 0 et 1"
    assert abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) < 0.001, \
        "TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT doit égaler 1.0"

    assert BATCH_SIZE > 0, "BATCH_SIZE doit être > 0"
    assert LEARNING_RATE > 0, "LEARNING_RATE doit être > 0"
    assert NUM_EPOCHS > 0, "NUM_EPOCHS doit être > 0"

    print("✅ Toutes les constantes sont valides")


if __name__ == '__main__':
    # Valider et afficher les constantes
    validate_constants()

    print("\n" + "="*80)
    print("CONSTANTES DU PROJET")
    print("="*80)

    print(f"\n📊 DONNÉES:")
    print(f"  Timeframe: {TIMEFRAME_MINUTES} min")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Trim edges: {TRIM_EDGES}")
    print(f"  BTC candles: {BTC_CANDLES:,}")
    print(f"  ETH candles: {ETH_CANDLES:,}")

    print(f"\n📈 INDICATEURS:")
    print(f"  RSI period: {RSI_PERIOD}")
    print(f"  CCI period: {CCI_PERIOD}")
    print(f"  Bollinger period: {BOL_PERIOD} (±{BOL_NUM_STD}σ)")
    print(f"  MACD: {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")

    print(f"\n🤖 MODÈLE:")
    print(f"  Input: {NUM_INDICATORS} indicateurs × {SEQUENCE_LENGTH} timesteps")
    print(f"  Output: {NUM_OUTPUTS} sorties (multi-output)")
    print(f"  CNN filters: {CNN_FILTERS}")
    print(f"  LSTM hidden: {LSTM_HIDDEN_SIZE} × {LSTM_NUM_LAYERS} layers")

    print(f"\n🎯 ENTRAÎNEMENT:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Split: {TRAIN_SPLIT:.0%} train / {VAL_SPLIT:.0%} val / {TEST_SPLIT:.0%} test")
