#!/bin/bash
# Script de comparaison: Proposition A (Smart Hybrid) vs Proposition B (Profitability)

INDICATOR=${1:-macd}

echo "=========================================================================="
echo "COMPARAISON: Proposition A vs Proposition B ($INDICATOR)"
echo "=========================================================================="
echo ""

echo "📊 TEST 1: Smart Hybrid Relabeling (Proposition A)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Règles:"
echo "  - Durée 3:    SUPPRIMER TOUT"
echo "  - Durée 4-5:  SUPPRIMER SI Vol Q4"
echo ""
python tests/test_smart_hybrid_relabeling.py --indicator $INDICATOR
echo ""
echo ""

echo "📊 TEST 2: Profitability Relabeling (Proposition B) - HORIZON 12"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Règles:"
echo "  - Regarder Max Return sur 12 bougies (1h)"
echo "  - Si Max Return < 0.2% → Relabeler WEAK"
echo ""
python tests/test_profitability_relabeling.py --indicator $INDICATOR --horizon 12 --fees 0.002
echo ""
echo ""

echo "📊 TEST 3: Profitability Relabeling (Proposition B) - HORIZON 6"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Règles:"
echo "  - Regarder Max Return sur 6 bougies (30 min)"
echo "  - Si Max Return < 0.2% → Relabeler WEAK"
echo ""
python tests/test_profitability_relabeling.py --indicator $INDICATOR --horizon 6 --fees 0.002
echo ""
echo ""

echo "=========================================================================="
echo "FIN DES TESTS COMPARATIFS"
echo "=========================================================================="
echo ""
echo "Analysez:"
echo "  - Proposition A: Compromis entre Config 3 et 4"
echo "  - Proposition B: Nettoyage basé sur vérité terrain (PnL futur)"
echo "  - Comparez ΔWin Rate, ΔPnL, ΔPrédictivité"
echo ""
