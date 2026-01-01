#!/bin/bash

# Script d'exemple pour lancer le pipeline de données
# Usage: bash example_run.sh

echo "🚀 Pipeline de Données Crypto - Exemple d'exécution"
echo "=================================================="
echo ""

# Créer les dossiers si nécessaires
mkdir -p data/processed

# Vérifier que les données sources existent
if [ ! -f "../data_trad/BTCUSD_all_5m.csv" ]; then
    echo "❌ Erreur: Fichier ../data_trad/BTCUSD_all_5m.csv introuvable"
    echo "   Assurez-vous que les données sont dans ../data_trad/"
    exit 1
fi

echo "✅ Données sources trouvées"
echo ""

# Lancer le pipeline pour BTC
echo "📊 Traitement BTC..."
python src/data_pipeline.py \
    --input ../data_trad/BTCUSD_all_5m.csv \
    --output data/processed/btc_30m_dataset.csv \
    --timeframe 30T \
    --label-source rsi \
    --smoothing 0.25

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset BTC généré avec succès!"
    echo "   Fichier: data/processed/btc_30m_dataset.csv"
else
    echo ""
    echo "❌ Erreur lors de la génération du dataset BTC"
    exit 1
fi

# Lancer le pipeline pour ETH (optionnel)
if [ -f "../data_trad/ETHUSD_all_5m.csv" ]; then
    echo ""
    echo "📊 Traitement ETH..."
    python src/data_pipeline.py \
        --input ../data_trad/ETHUSD_all_5m.csv \
        --output data/processed/eth_30m_dataset.csv \
        --timeframe 30T \
        --label-source rsi \
        --smoothing 0.25

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Dataset ETH généré avec succès!"
        echo "   Fichier: data/processed/eth_30m_dataset.csv"
    else
        echo ""
        echo "⚠️  Avertissement: Erreur lors de la génération du dataset ETH"
    fi
fi

echo ""
echo "=================================================="
echo "🎉 Pipeline terminé!"
echo ""
echo "Prochaines étapes:"
echo "1. Valider les datasets: jupyter notebook notebooks/01_data_validation.ipynb"
echo "2. Vérifier qu'il n'y a pas de data leakage"
echo "3. Commencer l'entraînement du modèle (Phase 2)"
echo ""
