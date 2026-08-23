"""Quick audit: verify min(k)==2 in all three regression panels."""
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

NL_STRUCTURAL_CHARS = set(',.;:?!\'"()[]{}')
SYM_COV_THRESHOLD = 0.50
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
K_ALL = [1] + K_VALUES
MIN_N = 30

def build_ngram(t, k):
    c = defaultdict(Counter)
    for i in range(k, len(t)):
        c[t[i-k:i]][t[i]] += 1
    return c

def compute_means(counts, k, test_text, vocab_size):
    res = defaultdict(lambda: {'sum_s': 0.0, 'n': 0, 'ctx_hit': 0})
    for i in range(k, len(test_text)):
        ch = test_text[i]
        if not (ch in NL_STRUCTURAL_CHARS or ch.isalpha() or ch.isdigit()):
            continue
        ctx = test_text[i-k:i]
        cc = counts.get(ctx, {})
        ct = sum(cc.values())
        prob = (cc.get(ch, 0) + 1) / (ct + vocab_size)
        r = res[ch]
        r['sum_s'] += -np.log2(prob)
        r['n'] += 1
        if cc:
            r['ctx_hit'] += 1
    return {ch: dict(v) for ch, v in res.items()}

def build_panel(text, name):
    n = len(text)
    n_tr = int(0.8 * n)
    n_vl = int(0.1 * n)
    train, test = text[:n_tr], text[n_tr+n_vl:]
    vocab_size = len(set(text))
    train_freq = Counter(train)
    total_train = len(train)

    from pathlib import Path
    print(f'\n--- {name} ---')
    S = {}
    for k in K_ALL:
        counts = build_ngram(train, k)
        S[k] = compute_means(counts, k, test, vocab_size)
        print(f'  k={k}: {len(S[k])} chars', flush=True)

    rows = []
    S1 = S[1]
    for ch, e1 in S1.items():
        if e1['n'] < MIN_N:
            continue
        if ch in NL_STRUCTURAL_CHARS:
            sym_type = 'structural'
        elif ch.isalpha() or ch.isdigit():
            sym_type = 'lexical'
        else:
            continue
        s1 = e1['sum_s'] / e1['n']
        freq = train_freq.get(ch, 1) / total_train
        for k in K_VALUES:
            ek = S[k].get(ch)
            if ek is None or ek['n'] < MIN_N:
                continue
            sym_cov = ek['ctx_hit'] / ek['n']
            if sym_cov < SYM_COV_THRESHOLD:
                continue
            rows.append({'ch': ch, 'k': k, 'type': sym_type})

    df = pd.DataFrame(rows)
    if df.empty:
        print('  EMPTY panel')
        return
    print(f'  Regression panel rows: {len(df)}')
    print(f'  min(k)={df.k.min()}  max(k)={df.k.max()}  k=1 rows: {(df.k==1).sum()}')
    print('  k counts:', df.k.value_counts().sort_index().to_dict())

# NL1
text1 = open('corpus_shakespeare.txt', encoding='utf-8', errors='replace').read()
build_panel(text1, 'NL1 shakespeare')

# NL2
raw = open('corpus_pride_prejudice.txt', encoding='utf-8', errors='replace').read()
s = raw.find('It is a truth')
e = raw.rfind('End of the Project Gutenberg')
text2 = raw[s:e] if s > 0 else raw
build_panel(text2, 'NL2 pride_prej')

print('\nDone. k=1 is excluded from all panels if k=1 rows == 0.')
