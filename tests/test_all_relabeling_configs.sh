#!/bin/bash
#
# Test toutes les configurations de relabeling pour trouver le sweet spot
# Volume vs Qualité
#

INDICATOR=${1:-macd}

echo "=========================================================================="
echo "TEST COMPARATIF - Configurations Relabeling ($INDICATOR)"
echo "=========================================================================="
echo ""

echo "📊 CONFIG 1: Duration 3 UNIQUEMENT (Conservateur)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python tests/test_relabeling_impact.py --indicator $INDICATOR --duration-trap 3
echo ""

echo "📊 CONFIG 2: Duration 3-4 (Compromis)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python tests/test_relabeling_impact.py --indicator $INDICATOR --duration-trap 3 4
echo ""

echo "📊 CONFIG 3: Duration 3-4-5 (Agressif - actuel)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python tests/test_relabeling_impact.py --indicator $INDICATOR --duration-trap 3 4 5
echo ""

echo "📊 CONFIG 4: Duration 3-4-5 AND Vol Q4 (Très conservateur)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python tests/test_relabeling_impact.py --indicator $INDICATOR --duration-trap 3 4 5 --vol-conditional
echo ""

echo "=========================================================================="
echo "FIN DES TESTS COMPARATIFS"
echo "=========================================================================="
echo ""
echo "Analysez:"
echo "  - ΔWin Rate (doit être positif)"
echo "  - ΔPnL Total (doit être positif ou faiblement négatif)"
echo "  - Prédictivité STRONG (plus élevé = meilleur)"
echo "  - Profit Factor (plus élevé = meilleur)"
echo ""
echo "Sweet spot = Compromis entre qualité (WR, PF, Prédictivité) et volume (PnL Total)"
