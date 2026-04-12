"""
Context formatter for Qwen — converts features + portfolio state into
structured text that Qwen can analyze.

Key insight: Qwen respects JSON output schema ONLY when input is natural text.
JSON input contaminates the model. This module produces the text input.
"""

from typing import Any, Dict, List, Optional


def format_user_message(
    features: dict,
    portfolio_state: dict,
    recent_trades: List[dict],
    session_stats: dict,
) -> str:
    """Build the complete user message for Qwen.

    Args:
        features: anonymized features dict from compute_features + anonymize_and_format
        portfolio_state: {positions: [...], cash: float, exposure_pct: float, equity: float}
        recent_trades: last 5 closed trades [{symbol, exit_reason, pnl_pct, duration_min}, ...]
        session_stats: {pnl_pct: float, n_closed: int, win_rate_pct: float}

    Returns:
        Structured text < 6000 tokens.
    """
    lines = []

    # --- CYCLE ---
    ci = features.get("cycle_index", "?")
    lines.append(f"CYCLE {ci}")

    # --- GLOBAL ---
    lines.append("")
    lines.append("# GLOBAL")
    g = features.get("global", {})
    btc = g.get("btc", {})
    lines.append(
        f"BTC: chg_1h={btc.get('chg_1h_pct')}% chg_4h={btc.get('chg_4h_pct')}% "
        f"chg_12h={btc.get('chg_12h_pct')}% chg_24h={btc.get('chg_24h_pct')}% "
        f"chg_7d={btc.get('chg_7d_pct')}%"
    )
    dom = g.get("btc_dominance", {})
    if isinstance(dom, dict) and dom.get("value") is not None:
        lines.append(f"Dominance: {dom['value']}% (chg_24h={dom.get('chg_24h_pct', 0)}%)")
    t = g.get("time", {})
    lines.append(f"Time to close: {t.get('minutes_to_close')} min")

    # --- CROSS-ASSET OVERVIEW ---
    lines.append("")
    lines.append("# CROSS-ASSET OVERVIEW")
    for a in features.get("assets", []):
        asset_id = a.get("id", a.get("_symbol", "?"))
        c1h = a.get("chg_1h_pct")
        c24h = a.get("chg_24h_pct")
        c7d = a.get("chg_7d_pct")
        lines.append(f"{asset_id}: 1h={c1h}% 24h={c24h}% 7d={c7d}%")

    # --- PORTFOLIO ---
    lines.append("")
    lines.append("# PORTFOLIO")
    positions = portfolio_state.get("positions", [])
    lines.append(f"Open positions ({len(positions)}):")
    if positions:
        for p in positions:
            lines.append(
                f"  - {p['symbol']}: entry={p['entry_price']:.4f} "
                f"current={p['current_price']:.4f} "
                f"P&L={p['pnl_pct']:+.2f}% "
                f"stop={p['stop_price']:.4f} tp={p['tp_price']:.4f} "
                f"age={p['age_minutes']}min"
            )
    else:
        lines.append("  (none)")
    cash = portfolio_state.get("cash", 0)
    exposure = portfolio_state.get("exposure_pct", 0)
    lines.append(f"Cash: ${cash:,.2f} | Exposure: {exposure:.1f}%")

    # --- RECENT TRADES ---
    lines.append("")
    lines.append("# RECENT TRADES (last 5 closed today)")
    if recent_trades:
        for rt in recent_trades[-5:]:
            lines.append(
                f"  - {rt['symbol']} buy -> {rt['exit_reason']}, "
                f"P&L={rt['pnl_pct']:+.2f}%, duration={rt['duration_min']}min"
            )
    else:
        lines.append("  (none)")

    # --- SESSION TODAY ---
    lines.append("")
    lines.append("# SESSION TODAY")
    lines.append(
        f"P&L: {session_stats.get('pnl_pct', 0):+.2f}% | "
        f"Trades closed: {session_stats.get('n_closed', 0)} | "
        f"Win rate: {session_stats.get('win_rate_pct', 0):.0f}%"
    )

    # --- ASSETS ---
    lines.append("")
    lines.append("# ASSETS")
    for a in features.get("assets", []):
        asset_id = a.get("id", a.get("_symbol", "?"))
        lines.append("")
        lines.append(f"{asset_id}:")
        lines.append(f"  price={a.get('price')}")

        # Session
        s = a.get("session", {})
        s_parts = []
        if s.get("chg_pct") is not None:
            s_parts.append(f"chg={s['chg_pct']}%")
        if s.get("range_position") is not None:
            s_parts.append(f"range_pos={s['range_position']}")
        if s_parts:
            lines.append(f"  session: {' '.join(s_parts)}")

        # Trend
        tr = a.get("trend", {})
        t_parts = []
        for ema_key, ema_data in tr.items():
            if isinstance(ema_data, dict) and ema_data.get("slope_pct") is not None:
                t_parts.append(
                    f"{ema_key}_slope={ema_data['slope_pct']}% dist={ema_data.get('dist_pct', '?')}%"
                )
        if t_parts:
            lines.append(f"  trend: {' | '.join(t_parts)}")

        # Regime
        r = a.get("regime", {})
        if r.get("adx_1h") is not None:
            lines.append(f"  regime: adx_1h={r['adx_1h']}")

        # Momentum
        m = a.get("momentum", {})
        m_parts = []
        if m.get("rsi_15m") is not None:
            m_parts.append(f"rsi_15m={m['rsi_15m']}")
        if m.get("rsi_1h") is not None:
            m_parts.append(f"rsi_1h={m['rsi_1h']}")
        if m_parts:
            lines.append(f"  momentum: {' '.join(m_parts)}")

        # Volatility
        v = a.get("volatility", {})
        v_parts = []
        if v.get("atr_15m_pct") is not None:
            v_parts.append(f"atr_pct={v['atr_15m_pct']}%")
        if v.get("atr_ratio_vs_avg50") is not None:
            v_parts.append(f"atr_ratio={v['atr_ratio_vs_avg50']}")
        if v.get("bb_position_15m") is not None:
            v_parts.append(f"bb_pos={v['bb_position_15m']}")
        if v_parts:
            lines.append(f"  volatility: {' '.join(v_parts)}")

        # Volume
        vol = a.get("volume", {})
        vol_parts = []
        if vol.get("vol_rel_15m") is not None:
            vol_parts.append(f"vol_rel_15m={vol['vol_rel_15m']}")
        if vol.get("vol_rel_1h") is not None:
            vol_parts.append(f"vol_rel_1h={vol['vol_rel_1h']}")
        if vol.get("vwap_dist_pct") is not None:
            vol_parts.append(f"vwap_dist={vol['vwap_dist_pct']}%")
        if vol_parts:
            lines.append(f"  volume: {' '.join(vol_parts)}")

        # Structure
        st = a.get("structure", {})
        sup = st.get("support", {})
        res = st.get("resistance", {})
        st_parts = []
        if sup.get("price") is not None:
            st_parts.append(f"support={sup['price']} ({sup.get('dist_pct','')}% {sup.get('type','')})")
        if res.get("price") is not None:
            st_parts.append(f"resistance={res['price']} ({res.get('dist_pct','')}% {res.get('type','')})")
        if st_parts:
            lines.append(f"  structure: {' / '.join(st_parts)}")

        # Correlation
        c = a.get("correlation", {})
        if c.get("corr_btc_24h") is not None:
            lines.append(f"  correlation: corr_btc_24h={c['corr_btc_24h']}")

        # Current bars
        cb = a.get("current_bar", {})
        for tf_key, cb_data in cb.items():
            if cb_data is not None and isinstance(cb_data, dict):
                cb_parts = []
                if cb_data.get("progress_pct") is not None:
                    cb_parts.append(f"progress={cb_data['progress_pct']}%")
                if cb_data.get("move_pct") is not None:
                    cb_parts.append(f"move={cb_data['move_pct']}%")
                if cb_data.get("range_vs_atr") is not None:
                    cb_parts.append(f"range_vs_atr={cb_data['range_vs_atr']}")
                if cb_data.get("vol_vs_expected") is not None:
                    cb_parts.append(f"vol_vs_exp={cb_data['vol_vs_expected']}")
                if cb_parts:
                    lines.append(f"  current_bar_{tf_key}: {' '.join(cb_parts)}")

        # Series 20 points
        series = a.get("series_20", {})
        if series:
            lines.append("  series_20:")
            for s_key, s_vals in series.items():
                if s_vals:
                    formatted = [f"{v:.4f}" if v is not None else "?" for v in s_vals]
                    lines.append(f"    {s_key}: [{', '.join(formatted)}]")

    return "\n".join(lines)
