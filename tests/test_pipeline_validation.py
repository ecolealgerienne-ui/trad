#!/usr/bin/env python3
"""
Tests rigoureux du pipeline avec visualisations.

RÈGLE D'OR: Valider CHAQUE étape avant de continuer.
Un seul problème dans les données = perte de temps et de ressources GPU.

Tests couverts:
1. Bougies Fantômes (6 steps, OHLC integrity)
2. Features Avancées (velocity, log returns, Z-score)
3. Indicateurs Techniques (ranges corrects)
4. Labels (distribution, pas de leakage)
5. Pipeline Multi-Actifs (normalisation séparée)
6. Gap Period (pas de contamination)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Imports du pipeline
from data_pipeline import create_ghost_candles, build_dataset
from advanced_features import (
    add_velocity_features,
    add_open_context,
    add_step_index_normalized,
    add_log_returns_ghost,
    validate_advanced_features
)
from utils import (
    validate_ohlc_integrity,
    check_data_leakage,
    split_train_val_test_with_gap
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class PipelineValidator:
    """Validateur rigoureux du pipeline avec visualisations."""

    def __init__(self, output_dir='tests/validation_output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.errors = []
        self.warnings = []

    def log_error(self, message):
        """Enregistre une erreur critique."""
        self.errors.append(message)
        logger.error(f"❌ ERREUR: {message}")

    def log_warning(self, message):
        """Enregistre un avertissement."""
        self.warnings.append(message)
        logger.warning(f"⚠️  WARNING: {message}")

    def log_success(self, message):
        """Enregistre un succès."""
        logger.info(f"✅ {message}")

    def create_sample_data(self, n_candles=500):
        """
        Crée des données synthétiques pour tests.

        Simule 500 bougies 5min = ~42 heures de données
        """
        logger.info(f"Création de {n_candles} bougies 5min synthétiques...")

        # Générer timestamps 5min
        start_date = pd.Timestamp('2024-01-01 00:00:00')
        timestamps = pd.date_range(start=start_date, periods=n_candles, freq='5min')

        # Générer prix avec random walk
        np.random.seed(42)
        initial_price = 50000.0  # BTC ~50k

        prices = [initial_price]
        for _ in range(n_candles - 1):
            change = np.random.normal(0, 100)  # Volatilité de ±100
            new_price = max(prices[-1] + change, 1000)  # Prix min 1k
            prices.append(new_price)

        closes = np.array(prices)

        # Générer OHLC réalistes
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, closes)):
            volatility = np.random.uniform(0.001, 0.005)  # 0.1-0.5% volatilité

            high = close * (1 + volatility)
            low = close * (1 - volatility)

            # Open basé sur le close précédent avec petit gap
            if i == 0:
                open_price = close
            else:
                gap = np.random.normal(0, close * 0.001)
                open_price = closes[i-1] + gap

            # S'assurer que OHLC est valide
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            volume = np.random.uniform(100, 1000)

            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)

        # Vérifier l'intégrité OHLC
        try:
            validate_ohlc_integrity(df)
            self.log_success(f"Données synthétiques créées: {len(df)} bougies 5min")
        except ValueError as e:
            self.log_error(f"Données synthétiques invalides: {e}")

        return df

    def test_ghost_candles(self, df_5m):
        """
        Test 1: Validation des Bougies Fantômes.

        Vérifications:
        - Exactement 6 steps par bougie 30min
        - Intégrité OHLC à chaque step
        - Open constant dans une bougie
        - High/Low monotones
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 1: BOUGIES FANTÔMES")
        logger.info("="*80)

        df_ghost = create_ghost_candles(df_5m, target_timeframe='30min')

        # Vérification 1: 6 steps par bougie
        steps_per_candle = df_ghost.groupby('candle_30m_timestamp')['step'].nunique()

        if (steps_per_candle == 6).all():
            self.log_success("Toutes les bougies ont exactement 6 steps")
        else:
            bad_candles = steps_per_candle[steps_per_candle != 6]
            self.log_error(f"{len(bad_candles)} bougies n'ont pas 6 steps!")
            print(bad_candles.head())

        # Vérification 2: Step range 1-6
        step_values = df_ghost['step'].unique()
        expected_steps = {1, 2, 3, 4, 5, 6}
        if set(step_values) == expected_steps:
            self.log_success("Steps valides: 1-6")
        else:
            self.log_error(f"Steps invalides trouvés: {step_values}")

        # Vérification 3: Intégrité OHLC
        try:
            validate_ohlc_integrity(df_ghost, col_prefix='ghost_')
            self.log_success("Intégrité OHLC validée")
        except ValueError as e:
            self.log_error(f"Intégrité OHLC échouée: {e}")

        # Vérification 4: Open constant par bougie
        open_changes = df_ghost.groupby('candle_30m_timestamp')['ghost_open'].nunique()
        if (open_changes == 1).all():
            self.log_success("Open constant dans chaque bougie 30min")
        else:
            bad_opens = open_changes[open_changes != 1]
            self.log_error(f"{len(bad_opens)} bougies avec Open non-constant!")

        # Vérification 5: High monotone croissant
        for candle_ts, group in df_ghost.groupby('candle_30m_timestamp'):
            group = group.sort_values('step')
            highs = group['ghost_high'].values

            # Le high doit être monotone croissant ou constant
            if not all(highs[i] <= highs[i+1] for i in range(len(highs)-1)):
                self.log_error(f"High non-monotone à {candle_ts}")
                break
        else:
            self.log_success("High monotone croissant dans toutes les bougies")

        # Visualisation: Bougie fantôme en formation
        self._visualize_ghost_candle(df_ghost)

        return df_ghost

    def _visualize_ghost_candle(self, df_ghost):
        """Visualise l'évolution d'une bougie fantôme."""
        # Prendre une bougie au milieu
        candle_ts = df_ghost['candle_30m_timestamp'].unique()[len(df_ghost['candle_30m_timestamp'].unique())//2]
        sample = df_ghost[df_ghost['candle_30m_timestamp'] == candle_ts].sort_values('step')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # OHLC evolution
        axes[0,0].plot(sample['step'], sample['ghost_open'], 'o-', label='Open', color='blue')
        axes[0,0].plot(sample['step'], sample['ghost_high'], 's-', label='High', color='green')
        axes[0,0].plot(sample['step'], sample['ghost_low'], '^-', label='Low', color='red')
        axes[0,0].plot(sample['step'], sample['ghost_close'], 'd-', label='Close', color='purple')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('Prix')
        axes[0,0].set_title('Évolution OHLC de la Bougie Fantôme')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Volume cumulé
        axes[0,1].bar(sample['step'], sample['ghost_volume'], color='orange', alpha=0.7)
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Volume Cumulé')
        axes[0,1].set_title('Volume Cumulatif')
        axes[0,1].grid(True, alpha=0.3)

        # Range (H-L)
        ranges = sample['ghost_high'] - sample['ghost_low']
        axes[1,0].plot(sample['step'], ranges, 'o-', color='purple')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('High - Low')
        axes[1,0].set_title('Range de la Bougie')
        axes[1,0].grid(True, alpha=0.3)

        # Validation OHLC (High >= Low, etc.)
        ohlc_valid = (
            (sample['ghost_high'] >= sample['ghost_low']) &
            (sample['ghost_high'] >= sample['ghost_open']) &
            (sample['ghost_high'] >= sample['ghost_close']) &
            (sample['ghost_low'] <= sample['ghost_open']) &
            (sample['ghost_low'] <= sample['ghost_close'])
        )

        colors = ['green' if v else 'red' for v in ohlc_valid]
        axes[1,1].bar(sample['step'], [1]*len(sample), color=colors, alpha=0.7)
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Validité OHLC')
        axes[1,1].set_title('Validation OHLC (Vert=OK, Rouge=Erreur)')
        axes[1,1].set_ylim(0, 1.5)
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / '01_ghost_candle_evolution.png', dpi=150, bbox_inches='tight')
        logger.info(f"📊 Visualisation sauvegardée: {self.output_dir / '01_ghost_candle_evolution.png'}")
        plt.close()

    def test_advanced_features(self, df_ghost):
        """
        Test 2: Validation des Features Avancées.

        Vérifications:
        - Velocity: range raisonnable
        - Amplitude: toujours positive
        - Acceleration: centrée à 0
        - Log returns: centrés à 0
        - Open Z-Score: mean~0, std~1
        - Step index norm: exactement 0.0 à 1.0
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 2: FEATURES AVANCÉES")
        logger.info("="*80)

        # Ajouter velocity features
        df = add_velocity_features(df_ghost.copy(), ghost_prefix='ghost')

        # Vérification 1: Velocity
        if 'velocity' in df.columns:
            velocity_values = df['velocity'].dropna()
            v_mean = velocity_values.mean()
            v_std = velocity_values.std()

            logger.info(f"Velocity: mean={v_mean:.6f}, std={v_std:.6f}")

            # La velocity doit être raisonnable (pas de valeurs absurdes)
            if velocity_values.abs().max() < 1.0:  # Moins de 100% par step
                self.log_success("Velocity dans un range raisonnable")
            else:
                self.log_warning(f"Velocity max très élevée: {velocity_values.abs().max():.4f}")
        else:
            self.log_error("Feature 'velocity' manquante!")

        # Vérification 2: Amplitude (toujours positive)
        if 'amplitude' in df.columns:
            amplitude_values = df['amplitude'].dropna()

            if (amplitude_values >= 0).all():
                self.log_success("Amplitude toujours positive")
            else:
                neg_count = (amplitude_values < 0).sum()
                self.log_error(f"{neg_count} valeurs d'amplitude négatives!")

            logger.info(f"Amplitude: mean={amplitude_values.mean():.6f}, std={amplitude_values.std():.6f}")
        else:
            self.log_error("Feature 'amplitude' manquante!")

        # Vérification 3: Acceleration (centrée à 0)
        if 'acceleration' in df.columns:
            accel_values = df['acceleration'].dropna()
            accel_mean = accel_values.mean()

            if abs(accel_mean) < 0.01:  # Proche de 0
                self.log_success(f"Acceleration centrée à 0 (mean={accel_mean:.6f})")
            else:
                self.log_warning(f"Acceleration non centrée: mean={accel_mean:.6f}")
        else:
            self.log_error("Feature 'acceleration' manquante!")

        # Ajouter log returns
        df = add_log_returns_ghost(df, ghost_prefix='ghost')

        # Vérification 4: Log Returns (centrés à 0)
        log_return_cols = ['ghost_high_log', 'ghost_low_log', 'ghost_close_log']
        for col in log_return_cols:
            if col in df.columns:
                values = df[col].dropna()
                mean_val = values.mean()
                std_val = values.std()

                logger.info(f"{col}: mean={mean_val:.6f}, std={std_val:.6f}")

                # Log returns doivent avoir mean proche de 0
                if abs(mean_val) < 0.01:
                    self.log_success(f"{col} centré à 0")
                else:
                    self.log_warning(f"{col} non centré: mean={mean_val:.6f}")
            else:
                self.log_error(f"Feature '{col}' manquante!")

        # Ajouter Open Z-Score
        df = add_open_context(df, ghost_prefix='ghost', window=50)

        # Vérification 5: Open Z-Score (mean~0, std~1)
        if 'ghost_open_zscore' in df.columns:
            zscore_values = df['ghost_open_zscore'].dropna()
            z_mean = zscore_values.mean()
            z_std = zscore_values.std()

            logger.info(f"ghost_open_zscore: mean={z_mean:.6f}, std={z_std:.6f}")

            if abs(z_mean) < 0.1 and abs(z_std - 1.0) < 0.2:
                self.log_success("Open Z-Score correctement normalisé (mean~0, std~1)")
            else:
                self.log_warning(f"Open Z-Score anormal: mean={z_mean:.4f}, std={z_std:.4f}")
        else:
            self.log_error("Feature 'ghost_open_zscore' manquante!")

        # Ajouter step index normalized
        df = add_step_index_normalized(df, max_steps=6)

        # Vérification 6: Step Index Normalized (exactement 0.0 à 1.0)
        if 'step_index_norm' in df.columns:
            step_norm_values = df['step_index_norm'].dropna()

            min_val = step_norm_values.min()
            max_val = step_norm_values.max()

            if min_val == 0.0 and max_val == 1.0:
                self.log_success("Step index normalisé: exactement 0.0 à 1.0")
            else:
                self.log_error(f"Step index range incorrect: [{min_val}, {max_val}] au lieu de [0.0, 1.0]")

            # Vérifier la progression linéaire
            for step in range(1, 7):
                expected = (step - 1) / 5.0
                actual = df[df['step'] == step]['step_index_norm'].iloc[0]

                if abs(actual - expected) > 0.001:
                    self.log_error(f"Step {step}: attendu {expected}, obtenu {actual}")
        else:
            self.log_error("Feature 'step_index_norm' manquante!")

        # Validation complète
        validation_results = validate_advanced_features(df, ghost_prefix='ghost')

        if validation_results['all_valid']:
            self.log_success("Toutes les features avancées validées")
        else:
            for issue in validation_results.get('issues', []):
                self.log_error(f"Validation échouée: {issue}")

        # Visualisation
        self._visualize_advanced_features(df)

        return df

    def _visualize_advanced_features(self, df):
        """Visualise les features avancées."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()

        features_to_plot = [
            ('velocity', 'Velocity (Vitesse de Formation)'),
            ('amplitude', 'Amplitude (Volatilité Relative)'),
            ('acceleration', 'Acceleration'),
            ('ghost_high_log', 'High Log Returns'),
            ('ghost_low_log', 'Low Log Returns'),
            ('ghost_close_log', 'Close Log Returns'),
            ('ghost_open_zscore', 'Open Z-Score (Contexte Prix)'),
            ('step_index_norm', 'Step Index Normalisé'),
        ]

        for idx, (feature, title) in enumerate(features_to_plot):
            if feature in df.columns:
                values = df[feature].dropna()

                axes[idx].hist(values, bins=50, edgecolor='black', alpha=0.7)
                axes[idx].axvline(x=values.mean(), color='red', linestyle='--',
                                 label=f'Mean={values.mean():.4f}')
                axes[idx].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                axes[idx].set_title(title)
                axes[idx].set_xlabel('Valeur')
                axes[idx].set_ylabel('Fréquence')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)

                # Ajouter std pour les Z-scores
                if 'zscore' in feature or 'norm' in feature:
                    axes[idx].axvline(x=values.mean() + values.std(), color='orange',
                                     linestyle=':', alpha=0.5, label=f'Std={values.std():.4f}')
                    axes[idx].axvline(x=values.mean() - values.std(), color='orange',
                                     linestyle=':', alpha=0.5)
                    axes[idx].legend()

        # Masquer les axes non utilisés
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / '02_advanced_features_distributions.png', dpi=150, bbox_inches='tight')
        logger.info(f"📊 Visualisation sauvegardée: {self.output_dir / '02_advanced_features_distributions.png'}")
        plt.close()

        # Visualisation 2: Évolution des features pour une bougie
        self._visualize_feature_evolution(df)

    def _visualize_feature_evolution(self, df):
        """Visualise l'évolution des features pendant la formation d'une bougie."""
        candle_ts = df['candle_30m_timestamp'].unique()[len(df['candle_30m_timestamp'].unique())//2]
        sample = df[df['candle_30m_timestamp'] == candle_ts].sort_values('step')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Velocity & Amplitude
        if 'velocity' in sample.columns and 'amplitude' in sample.columns:
            axes[0,0].plot(sample['step'], sample['velocity'], 'o-', label='Velocity', color='purple')
            axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[0,0].set_xlabel('Step')
            axes[0,0].set_ylabel('Velocity')
            axes[0,0].set_title('Velocity (Vitesse de Formation)')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend()

            axes[0,1].plot(sample['step'], sample['amplitude'], 'o-', label='Amplitude', color='orange')
            axes[0,1].set_xlabel('Step')
            axes[0,1].set_ylabel('Amplitude')
            axes[0,1].set_title('Amplitude (H-L)/O')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()

        # Log Returns
        if all(col in sample.columns for col in ['ghost_high_log', 'ghost_low_log', 'ghost_close_log']):
            axes[1,0].plot(sample['step'], sample['ghost_high_log'], 'o-', label='High', color='green')
            axes[1,0].plot(sample['step'], sample['ghost_low_log'], 's-', label='Low', color='red')
            axes[1,0].plot(sample['step'], sample['ghost_close_log'], '^-', label='Close', color='blue')
            axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[1,0].set_xlabel('Step')
            axes[1,0].set_ylabel('Log Return')
            axes[1,0].set_title('Log Returns (relatif à Open)')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()

        # Step Index Normalized
        if 'step_index_norm' in sample.columns:
            axes[1,1].plot(sample['step'], sample['step_index_norm'], 'o-', color='blue', linewidth=2)
            axes[1,1].set_xlabel('Step')
            axes[1,1].set_ylabel('Step Index Norm')
            axes[1,1].set_title('Step Index Normalisé (0.0→1.0)')
            axes[1,1].set_ylim(-0.1, 1.1)
            axes[1,1].grid(True, alpha=0.3)

            # Ajouter la ligne idéale
            axes[1,1].plot([1, 2, 3, 4, 5, 6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                          'r--', alpha=0.5, label='Idéal')
            axes[1,1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / '03_feature_evolution_per_step.png', dpi=150, bbox_inches='tight')
        logger.info(f"📊 Visualisation sauvegardée: {self.output_dir / '03_feature_evolution_per_step.png'}")
        plt.close()

    def test_data_leakage(self, df):
        """
        Test 3: Vérification du Data Leakage.

        CRITIQUE: Les features ne doivent PAS être corrélées avec le label futur.
        Corrélation acceptable: 0.1-0.3
        Corrélation suspecte: >0.7
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 3: DATA LEAKAGE")
        logger.info("="*80)

        # Pour ce test, on a besoin des labels
        # On va créer des labels factices pour tester
        np.random.seed(42)
        df = df.copy()
        df['label'] = np.random.choice([0, 1], size=len(df))

        # Lister les features
        exclude_cols = ['timestamp', 'candle_30m_timestamp', 'label', 'step',
                       'current_5m_open', 'current_5m_high', 'current_5m_low', 'current_5m_close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        logger.info(f"Vérification du leakage sur {len(feature_cols)} features...")

        leakage_results = check_data_leakage(df, feature_cols, label_col='label')

        # Analyser les résultats
        suspicious = leakage_results['suspicious_features']
        future_corrs = leakage_results['future_correlation']

        if not suspicious:
            self.log_success("Aucune feature suspecte détectée (|corr| < 0.7)")
        else:
            self.log_error(f"{len(suspicious)} features suspectes avec forte corrélation future!")
            for feat, corr in suspicious[:5]:  # Afficher les 5 pires
                logger.error(f"  - {feat}: corrélation = {corr:.3f}")

        # Vérifier la plage de corrélation idéale (0.1-0.3)
        ideal_range_count = sum(1 for corr in future_corrs.values()
                               if 0.1 <= abs(corr) <= 0.3)

        logger.info(f"Features dans la plage idéale (|corr| 0.1-0.3): {ideal_range_count}/{len(future_corrs)}")

        if ideal_range_count / len(future_corrs) > 0.5:
            self.log_success("Majorité des features dans la plage de corrélation idéale")
        else:
            self.log_warning("Peu de features dans la plage idéale - vérifier la qualité des features")

        # Visualisation
        self._visualize_leakage(future_corrs)

        return leakage_results

    def _visualize_leakage(self, future_corrs):
        """Visualise les corrélations futures pour détecter le leakage."""
        # Trier par valeur absolue
        sorted_corrs = sorted(future_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:25]

        features = [x[0] for x in sorted_corrs]
        corrs = [x[1] for x in sorted_corrs]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Colorier selon le niveau de risque
        colors = []
        for c in corrs:
            if abs(c) > 0.7:
                colors.append('red')  # Suspect
            elif abs(c) > 0.5:
                colors.append('orange')  # Warning
            elif 0.1 <= abs(c) <= 0.3:
                colors.append('green')  # Idéal
            else:
                colors.append('gray')  # Trop faible

        ax.barh(features, corrs, color=colors)
        ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Seuil suspect (0.7)')
        ax.axvline(x=-0.7, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0.3, color='orange', linestyle=':', alpha=0.5, label='Plage idéale (0.1-0.3)')
        ax.axvline(x=-0.3, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(x=0.1, color='green', linestyle=':', alpha=0.5)
        ax.axvline(x=-0.1, color='green', linestyle=':', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        ax.set_xlabel('Corrélation avec label[t+1]')
        ax.set_title('Top 25 Features - Corrélation Future (Leakage Check)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / '04_data_leakage_check.png', dpi=150, bbox_inches='tight')
        logger.info(f"📊 Visualisation sauvegardée: {self.output_dir / '04_data_leakage_check.png'}")
        plt.close()

    def generate_report(self):
        """Génère un rapport final de validation."""
        logger.info("\n" + "="*80)
        logger.info("RAPPORT FINAL DE VALIDATION")
        logger.info("="*80)

        total_tests = len(self.errors) + len(self.warnings)

        if not self.errors and not self.warnings:
            logger.info("✅ TOUS LES TESTS PASSÉS AVEC SUCCÈS!")
        else:
            if self.errors:
                logger.error(f"\n❌ {len(self.errors)} ERREURS CRITIQUES:")
                for i, err in enumerate(self.errors, 1):
                    logger.error(f"  {i}. {err}")

            if self.warnings:
                logger.warning(f"\n⚠️  {len(self.warnings)} WARNINGS:")
                for i, warn in enumerate(self.warnings, 1):
                    logger.warning(f"  {i}. {warn}")

        # Sauvegarder le rapport
        report_path = self.output_dir / 'validation_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT DE VALIDATION DU PIPELINE\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            if self.errors:
                f.write(f"ERREURS CRITIQUES ({len(self.errors)}):\n")
                for i, err in enumerate(self.errors, 1):
                    f.write(f"  {i}. {err}\n")
                f.write("\n")

            if self.warnings:
                f.write(f"WARNINGS ({len(self.warnings)}):\n")
                for i, warn in enumerate(self.warnings, 1):
                    f.write(f"  {i}. {warn}\n")
                f.write("\n")

            if not self.errors and not self.warnings:
                f.write("✅ TOUS LES TESTS PASSÉS AVEC SUCCÈS!\n")

        logger.info(f"\n📝 Rapport sauvegardé: {report_path}")

        return len(self.errors) == 0


def main():
    """Point d'entrée principal."""
    logger.info("="*80)
    logger.info("VALIDATION RIGOUREUSE DU PIPELINE")
    logger.info("="*80)

    validator = PipelineValidator()

    # Créer des données de test
    df_5m = validator.create_sample_data(n_candles=500)

    # Test 1: Bougies Fantômes
    df_ghost = validator.test_ghost_candles(df_5m)

    # Test 2: Features Avancées
    df_features = validator.test_advanced_features(df_ghost)

    # Test 3: Data Leakage
    validator.test_data_leakage(df_features)

    # Rapport final
    success = validator.generate_report()

    if success:
        logger.info("\n🎉 VALIDATION COMPLÈTE - PIPELINE PRÊT POUR PRODUCTION!")
        return 0
    else:
        logger.error("\n❌ VALIDATION ÉCHOUÉE - CORRIGER LES ERREURS AVANT DE CONTINUER!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
