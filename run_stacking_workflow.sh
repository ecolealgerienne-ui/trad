#!/bin/bash
################################################################################
# STACKING WORKFLOW - Pipeline Complet Automatisé
#
# Ce script automatise toutes les étapes du Stacking pour améliorer
# l'accuracy Direction de 92% → 95-96% en combinant les 3 experts.
#
# Objectif: Résoudre le Proxy Learning Failure (Win Rate 14% → 55-65%)
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🎯 STACKING WORKFLOW - Combinaison des 3 Experts (MACD, RSI, CCI)"
echo "================================================================================"
echo ""
echo "Objectif: Améliorer Direction Accuracy 92% → 95-96%"
echo "Hypothèse: Meilleure prédiction du Kalman → Win Rate 14% → 55-65%"
echo ""

# Configuration
ASSETS="BTC ETH BNB ADA LTC"
EPOCHS=50
DEVICE="cuda"  # ou cpu

################################################################################
# ÉTAPE 0: Vérification des Prérequis
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 ÉTAPE 0: Vérification des Prérequis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check 1: Datasets dual_binary
echo "🔍 Vérification datasets dual_binary..."
DATASETS_NEEDED=(
    "data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz"
    "data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz"
    "data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz"
)

DATASETS_MISSING=()
for dataset in "${DATASETS_NEEDED[@]}"; do
    if [ ! -f "$dataset" ]; then
        DATASETS_MISSING+=("$dataset")
        echo "   ❌ MANQUANT: $dataset"
    else
        echo "   ✅ TROUVÉ: $dataset"
    fi
done

# Check 2: Modèles entraînés
echo ""
echo "🔍 Vérification modèles entraînés..."
MODELS_NEEDED=(
    "models/best_model_macd_kalman_dual_binary.pth"
    "models/best_model_rsi_kalman_dual_binary.pth"
    "models/best_model_cci_kalman_dual_binary.pth"
)

MODELS_MISSING=()
for model in "${MODELS_NEEDED[@]}"; do
    if [ ! -f "$model" ]; then
        MODELS_MISSING+=("$model")
        echo "   ❌ MANQUANT: $model"
    else
        echo "   ✅ TROUVÉ: $model"
    fi
done

echo ""

################################################################################
# ÉTAPE 1: Génération des Datasets (si manquants)
################################################################################

if [ ${#DATASETS_MISSING[@]} -gt 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 ÉTAPE 1: Génération des Datasets Dual-Binary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "⚠️  ${#DATASETS_MISSING[@]} dataset(s) manquant(s)"
    echo ""
    echo "🚀 Commande:"
    echo "   python src/prepare_data_purified_dual_binary.py --assets $ASSETS"
    echo ""

    read -p "🤔 Générer les datasets maintenant? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "⏳ Génération en cours (durée estimée: ~5 min)..."
        python src/prepare_data_purified_dual_binary.py --assets $ASSETS

        if [ $? -eq 0 ]; then
            echo "✅ Datasets générés avec succès!"
        else
            echo "❌ ERREUR lors de la génération des datasets"
            exit 1
        fi
    else
        echo "⏭️  Skipped. Exécutez manuellement:"
        echo "   python src/prepare_data_purified_dual_binary.py --assets $ASSETS"
        echo ""
        echo "❌ Workflow interrompu (datasets manquants)"
        exit 1
    fi
else
    echo "✅ ÉTAPE 1: Tous les datasets existent déjà"
fi

echo ""

################################################################################
# ÉTAPE 2: Entraînement des 3 Modèles de Base (si manquants)
################################################################################

if [ ${#MODELS_MISSING[@]} -gt 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧠 ÉTAPE 2: Entraînement des 3 Modèles de Base"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "⚠️  ${#MODELS_MISSING[@]} modèle(s) manquant(s)"
    echo ""
    echo "🚀 Commandes:"
    echo "   python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz --epochs $EPOCHS"
    echo "   python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz --epochs $EPOCHS"
    echo "   python src/train.py --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz --epochs $EPOCHS"
    echo ""

    read -p "🤔 Entraîner les modèles maintenant? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "⏳ Entraînement MACD (durée estimée: ~10-30 min)..."
        python src/train.py \
            --data data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz \
            --epochs $EPOCHS \
            --device $DEVICE

        echo ""
        echo "⏳ Entraînement RSI (durée estimée: ~10-30 min)..."
        python src/train.py \
            --data data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz \
            --epochs $EPOCHS \
            --device $DEVICE

        echo ""
        echo "⏳ Entraînement CCI (durée estimée: ~10-30 min)..."
        python src/train.py \
            --data data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz \
            --epochs $EPOCHS \
            --device $DEVICE

        if [ $? -eq 0 ]; then
            echo "✅ Les 3 modèles entraînés avec succès!"
        else
            echo "❌ ERREUR lors de l'entraînement"
            exit 1
        fi
    else
        echo "⏭️  Skipped. Exécutez manuellement les 3 commandes ci-dessus"
        echo ""
        echo "❌ Workflow interrompu (modèles manquants)"
        exit 1
    fi
else
    echo "✅ ÉTAPE 2: Tous les modèles existent déjà"
fi

echo ""

################################################################################
# ÉTAPE 3: Génération des Méta-Features
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 ÉTAPE 3: Génération des Méta-Features"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 Objectif: Générer les prédictions des 3 modèles pour Train/Val/Test"
echo "📊 Output: X_meta (n, 6), Y_meta (n, 1) pour chaque split"
echo ""

# Check si méta-features existent déjà
if [ -f "data/meta/meta_features_train.npz" ] && \
   [ -f "data/meta/meta_features_val.npz" ] && \
   [ -f "data/meta/meta_features_test.npz" ]; then
    echo "⚠️  Les méta-features existent déjà"
    read -p "🤔 Régénérer? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "⏭️  Méta-features existantes réutilisées"
        echo ""
    else
        rm -rf data/meta/
        echo "🗑️  Anciennes méta-features supprimées"
    fi
fi

if [ ! -f "data/meta/meta_features_train.npz" ]; then
    echo "🚀 Commande:"
    echo "   python src/generate_meta_features.py --assets $ASSETS --device $DEVICE"
    echo ""
    echo "⏳ Génération en cours (durée estimée: ~2-3 min)..."

    python src/generate_meta_features.py --assets $ASSETS --device $DEVICE

    if [ $? -eq 0 ]; then
        echo "✅ Méta-features générées avec succès!"
        echo ""
        echo "📂 Fichiers créés:"
        ls -lh data/meta/*.npz
    else
        echo "❌ ERREUR lors de la génération des méta-features"
        exit 1
    fi
else
    echo "✅ Méta-features déjà disponibles"
fi

echo ""

################################################################################
# ÉTAPE 4: Entraînement du Meta-Modèle
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 ÉTAPE 4: Entraînement du Meta-Modèle (Stacking)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 Objectif: Apprendre à combiner les 3 experts pour retrouver le Kalman"
echo ""
echo "📋 Trois modèles disponibles (du plus simple au plus complexe):"
echo "   1. Logistic Regression (baseline, interprétable, ~10s)"
echo "   2. Random Forest (non-linéaire, feature importance, ~30s)"
echo "   3. MLP (neural network, patterns complexes, ~2 min)"
echo ""
echo "💡 Recommandation: Commencer par Logistic, puis tester RF/MLP si besoin"
echo ""

read -p "🤔 Quel modèle entraîner? [1=Logistic, 2=RF, 3=MLP, A=All] " -n 1 -r
echo ""

case $REPLY in
    1)
        MODELS_TO_TRAIN=("logistic")
        ;;
    2)
        MODELS_TO_TRAIN=("rf")
        ;;
    3)
        MODELS_TO_TRAIN=("mlp")
        ;;
    [Aa])
        MODELS_TO_TRAIN=("logistic" "rf" "mlp")
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac

for model_type in "${MODELS_TO_TRAIN[@]}"; do
    echo ""
    echo "⏳ Entraînement $model_type..."
    echo "🚀 Commande:"
    echo "   python src/train_stacking.py --model $model_type --device $DEVICE"
    echo ""

    python src/train_stacking.py --model $model_type --device $DEVICE

    if [ $? -eq 0 ]; then
        echo "✅ Modèle $model_type entraîné avec succès!"
    else
        echo "❌ ERREUR lors de l'entraînement de $model_type"
        exit 1
    fi
done

echo ""

################################################################################
# ÉTAPE 5: Résumé et Critères de Succès
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 RÉSUMÉ ET CRITÈRES DE SUCCÈS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Workflow Stacking terminé avec succès!"
echo ""
echo "📊 Résultats Meta-Modèle:"
echo "   → Consulter les logs ci-dessus pour les accuracy Train/Val/Test"
echo ""
echo "🎯 Critères de Succès:"
echo "   ✅ Test Accuracy ≥ 95% ?"
echo "   ✅ Gap Train/Test < 5% ?"
echo "   ✅ Amélioration vs Baseline (+3-4%) ?"
echo ""
echo "📋 Prochaines Étapes:"
echo ""
echo "1. SI 3/3 ✅ → Tester en backtest:"
echo "   python src/backtest_stacking.py"
echo ""
echo "2. SI Test Acc < 94% → Diagnostiquer:"
echo "   - Vérifier diversité des 3 modèles de base"
echo "   - Tester avec d'autres features (volatilité, volume)"
echo ""
echo "3. SI Overfit (gap > 5%) → Réduire complexité:"
echo "   - Revenir à Logistic ou RF"
echo "   - Augmenter dropout si MLP"
echo ""
echo "4. SI Succès → Combiner avec Profitability Relabeling:"
echo "   - Stacking pour Direction (92% → 95%)"
echo "   - Profitability pour Force (nettoyage STRONG)"
echo "   - Gain total attendu: Win Rate 14% → 65-70% 🏆"
echo ""
echo "📚 Documentation complète: STACKING_GUIDE.md"
echo ""
echo "================================================================================"
echo "🏁 FIN DU WORKFLOW STACKING"
echo "================================================================================"
