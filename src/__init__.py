"""
Pipeline de données pour prédiction de tendance crypto.

Modules:
- utils: Fonctions utilitaires communes
- filters: Filtres de signal (Octave, Savgol, etc.)
- indicators: Indicateurs techniques (RSI, CCI, MACD, Bollinger)
- normalization: Normalisation des features (Z-Score, Relative Open)
- labeling: Génération des labels de prédiction
- data_pipeline: Pipeline principal de transformation
"""

__version__ = '0.1.0'
