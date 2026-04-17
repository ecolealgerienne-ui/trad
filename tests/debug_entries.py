import json

with open('logs/test_run_20260412_224953.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        rec = json.loads(line)
        if not rec.get('success') or not rec.get('parsed'):
            continue
        g = rec['parsed'].get('global', {})
        max_pos = g.get('max_concurrent_positions')
        mode = g.get('market_mode')
        regime = g.get('btc_regime')
        print(f'Cycle {i}: max_pos={max_pos}, mode={mode}, regime={regime}')
        for a in rec['parsed'].get('assets', []):
            if a.get('action') == 'buy' and a.get('conviction', 0) >= 7:
                sym = a['symbol']
                conv = a['conviction']
                ez = a.get('entry_zone')
                sm = a.get('atr_stop_multiplier')
                print(f'  {sym}: buy conv={conv} entry_zone={ez} stop_mult={sm}')
        snap = rec.get('features_snapshot', {})
        for k, v in snap.items():
            print(f'  snapshot {k}: price={v.get("price")} atr={v.get("atr_15m_abs")}')
