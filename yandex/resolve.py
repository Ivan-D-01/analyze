import itertools
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


ALPHABET = ['F', 'W', 'E']
L = 5


# =========================
# 1. Загрузка и базовые проверки
# =========================

def load_data(train_path='yandex/train_runes.csv', test_path='yandex/test_runes.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def all_runes():
    return [''.join(p) for p in itertools.product(ALPHABET, repeat=L)]


def check_basic_consistency(train: pd.DataFrame):
    assert {'rune', 'spell'}.issubset(train.columns)
    assert train['rune'].map(len).eq(L).all(), "Есть строки не длины 5"
    assert train['rune'].map(lambda s: set(s) <= set(ALPHABET)).all(), "Есть посторонние символы"
    assert set(train['spell'].unique()) <= {0, 1}, "spell должен быть 0/1"

    dup = train.groupby('rune')['spell'].nunique().reset_index(name='nuniq')
    contradictions = dup[dup['nuniq'] > 1]

    n_total = len(train)
    n_unique = train['rune'].nunique()
    n_space = len(all_runes())
    covered = n_unique
    missing = sorted(set(all_runes()) - set(train['rune']))

    print('Rows in train:', n_total)
    print('Unique runes in train:', n_unique)
    print('Total possible runes:', n_space)
    print('Coverage:', f'{covered}/{n_space} = {covered / n_space:.3f}')
    print('Contradictory duplicate runes:', len(contradictions))
    if len(contradictions):
        print(contradictions)

    return {
        'n_total': n_total,
        'n_unique': n_unique,
        'n_space': n_space,
        'coverage': covered / n_space,
        'contradictions': contradictions,
        'missing': missing,
    }


# =========================
# 2. Признаки / гипотезы
# =========================

def rune_to_basic_features(r: str):
    c = Counter(r)
    runs = []
    curr = 1
    for i in range(1, L):
        if r[i] == r[i - 1]:
            curr += 1
        else:
            runs.append(curr)
            curr = 1
    runs.append(curr)
    max_run = max(runs)
    avg_run = sum(runs) / len(runs)
    transitions = sum(r[i] != r[i+1] for i in range(L - 1))
    first_pos = {ch: r.find(ch) for ch in ALPHABET}
    last_pos = {ch: r.rfind(ch) for ch in ALPHABET}
    islands = {}
    for ch in ALPHABET:
        cnt = 0
        for i in range(L):
            if r[i] == ch and (i == 0 or r[i - 1] != ch):
                cnt += 1
        islands[ch] = cnt
    n_unique = len(set(r))
    balance = max(c.values()) - min(c.values())
    feats = {
        'rune': r,
        'count_F': c['F'],
        'count_W': c['W'],
        'count_E': c['E'],
        'first': r[0],
        'last': r[-1],
        'middle': r[2],
        'fw_transitions': transitions,
        'transitions_ratio': transitions / (L - 1),
        'n_blocks': 1 + transitions,
        'has_FF': int('FF' in r),
        'has_WW': int('WW' in r),
        'has_EE': int('EE' in r),
        'has_FFF': int('FFF' in r),
        'has_WWW': int('WWW' in r),
        'has_EEE': int('EEE' in r),
        'has_FFFF': int('FFFF' in r),
        'has_WWWW': int('WWWW' in r),
        'has_EEEE': int('EEEE' in r),
        'is_palindrome': int(r == r[::-1]),
        'sorted_multiset': ''.join(sorted(r)),
        'max_run_len': max_run,
        'avg_run_len': avg_run,
        'first_pos_F': first_pos['F'],
        'first_pos_W': first_pos['W'],
        'first_pos_E': first_pos['E'],
        'last_pos_F': last_pos['F'],
        'last_pos_W': last_pos['W'],
        'last_pos_E': last_pos['E'],
        'span_pos_F': last_pos['F'] - first_pos['F'],
        'span_pos_W': last_pos['W'] - first_pos['W'],
        'span_pos_E': last_pos['E'] - first_pos['E'],
        'islands_F': islands['F'],
        'islands_W': islands['W'],
        'islands_E': islands['E'],
        'n_unique': n_unique,
        'balance_counts': balance,
        'dom_F_ge_3': int(c['F'] >= 3),
        'dom_W_ge_3': int(c['W'] >= 3),
        'dom_E_ge_3': int(c['E'] >= 3),
        'sym_pos0_eq_pos4': int(r[0] == r[4]),
        'sym_pos1_eq_pos3': int(r[1] == r[3]),
        'sym_pos0_eq_pos2': int(r[0] == r[2]),
        'sym_pos2_eq_pos4': int(r[2] == r[4]),
        'gap_pair02': r[0] + r[2],
        'gap_pair13': r[1] + r[3],
        'gap_pair24': r[2] + r[4],
    }
    for i, ch in enumerate(r):
        feats[f'pos{i}'] = ch
    for i in range(L - 1):
        feats[f'pair{i}{i+1}'] = r[i:i+2]
    for i in range(L - 2):
        feats[f'triplet{i}{i+1}{i+2}'] = r[i:i+3]
    return feats


def build_feature_df(df: pd.DataFrame):
    feats = pd.DataFrame([rune_to_basic_features(r) for r in df['rune']])
    return pd.concat([df.reset_index(drop=True), feats.drop(columns=['rune'])], axis=1)


def build_feature_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_feat = build_feature_df(train_df)
    test_feat = build_feature_df(test_df)

    y_train = train_feat['spell'].values

    # Разделим на числовые и категориальные признаки
    categorical_cols = []
    numeric_cols = []
    for col in train_feat.columns:
        if col in ['rune', 'spell']:
            continue
        if train_feat[col].dtype == object:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols),
        ]
    )

    X_train = preprocessor.fit_transform(train_feat.drop(columns=['rune', 'spell']))
    X_test = preprocessor.transform(test_feat.drop(columns=['rune']))

    return X_train, y_train, X_test


# Проверка "гипотеза верна" в строгом смысле:
# если две строки имеют одинаковые значения выбранных признаков,
# то у них должна быть одна и та же метка.
def hypothesis_is_exact(df: pd.DataFrame, cols):
    g = df.groupby(cols)['spell'].nunique()
    bad = g[g > 1]
    return len(bad) == 0, bad


def test_exact_hypotheses(train_feat: pd.DataFrame):
    hypotheses = {
        'counts_only': ['count_F', 'count_W', 'count_E'],
        'sorted_multiset_only': ['sorted_multiset'],
        'positions_only': [f'pos{i}' for i in range(L)],
        'first_last_counts': ['first', 'last', 'count_F', 'count_W', 'count_E'],
        'positions_plus_counts': [f'pos{i}' for i in range(L)] + ['count_F', 'count_W', 'count_E'],
        'adjacent_pairs_only': [f'pair{i}{i+1}' for i in range(L - 1)],
        'triplets_only': [f'triplet{i}{i+1}{i+2}' for i in range(L - 2)],
        'pairs_plus_counts': [f'pair{i}{i+1}' for i in range(L - 1)] + ['count_F', 'count_W', 'count_E'],
        'triplets_plus_counts': [f'triplet{i}{i+1}{i+2}' for i in range(L - 2)] + ['count_F', 'count_W', 'count_E'],
        'blocks_counts_endpoints': ['n_blocks', 'count_F', 'count_W', 'count_E', 'first', 'last'],
    }

    results = []
    for name, cols in hypotheses.items():
        ok, bad = hypothesis_is_exact(train_feat, cols)
        results.append((name, ok, len(bad)))
    res_df = pd.DataFrame(results, columns=['hypothesis', 'exact_on_train', 'n_conflicting_feature_patterns'])
    return res_df.sort_values(['exact_on_train', 'n_conflicting_feature_patterns'], ascending=[False, True])


# =========================
# 3. Проверка "насколько близки" простые модели
# =========================

def build_onehot_matrix(df: pd.DataFrame):
    X = np.zeros((len(df), L * len(ALPHABET)), dtype=int)
    idx = {'F': 0, 'W': 1, 'E': 2}
    for i, r in enumerate(df['rune']):
        for j, ch in enumerate(r):
            X[i, 3*j + idx[ch]] = 1
    return X


def train_accuracy_of_models(train: pd.DataFrame):
    y = train['spell'].values
    X_oh = build_onehot_matrix(train)

    models = {}

    # Линейная модель по позиционному one-hot
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_oh, y)
    models['logreg_onehot_train_acc'] = (lr.predict(X_oh) == y).mean()

    # Квадратичные взаимодействия по one-hot
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X2 = poly.fit_transform(X_oh)
    lr2 = LogisticRegression(max_iter=20000)
    lr2.fit(X2, y)
    models['logreg_onehot_pairwise_interactions_train_acc'] = (lr2.predict(X2) == y).mean()

    # Дерево разных глубин
    for depth in range(1, 8):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
        tree.fit(X_oh, y)
        models[f'decision_tree_depth_{depth}_train_acc'] = (tree.predict(X_oh) == y).mean()

    # Полное дерево
    tree_full = DecisionTreeClassifier(random_state=0)
    tree_full.fit(X_oh, y)
    models['decision_tree_full_train_acc'] = (tree_full.predict(X_oh) == y).mean()

    return models


def loo_accuracy(train: pd.DataFrame):
    y = train['spell'].values
    X_oh = build_onehot_matrix(train)

    loo = LeaveOneOut()
    res = {}

    for depth in [1, 2, 3, 4, 5, 6, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
        scores = cross_val_score(tree, X_oh, y, cv=loo, scoring='accuracy', n_jobs=None)
        name = f'decision_tree_depth_{depth}_loo_acc'
        res[name] = scores.mean()

    # Линейная модель
    lr = LogisticRegression(max_iter=10000)
    scores = cross_val_score(lr, X_oh, y, cv=loo, scoring='accuracy', n_jobs=None)
    res['logreg_onehot_loo_acc'] = scores.mean()

    # Логрег с попарными взаимодействиями
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X2 = poly.fit_transform(X_oh)
    lr2 = LogisticRegression(max_iter=20000)
    scores = cross_val_score(lr2, X2, y, cv=loo, scoring='accuracy', n_jobs=None)
    res['logreg_onehot_pairwise_interactions_loo_acc'] = scores.mean()

    return res


# =========================
# 4. Поиск полного правила деревом
# =========================

def fit_interpretable_tree(train: pd.DataFrame):
    y = train['spell'].values
    X = build_onehot_matrix(train)

    # Сначала ищем минимальную глубину, которая даёт train_acc = 1
    best_tree = None
    best_depth = None
    for depth in range(1, 11):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
        tree.fit(X, y)
        acc = (tree.predict(X) == y).mean()
        if acc == 1.0:
            best_tree = tree
            best_depth = depth
            break

    if best_tree is None:
        best_tree = DecisionTreeClassifier(random_state=0)
        best_tree.fit(X, y)
        best_depth = None

    feature_names = []
    for pos in range(L):
        for ch in ALPHABET:
            feature_names.append(f'pos{pos}=={ch}')

    rules = export_text(best_tree, feature_names=feature_names)
    return best_tree, best_depth, rules


# =========================
# 5. Применение модели ко всем 243 комбинациям и к test
# =========================

def predict_all_space(model):
    full = pd.DataFrame({'rune': all_runes()})
    X_full = build_onehot_matrix(full)
    full['pred'] = model.predict(X_full)
    return full


def make_submission(train_path='yandex/train_runes.csv', test_path='yandex/test_runes.csv', out_path='yandex/answers.csv'):
    train, test = load_data(train_path, test_path)

    X_train, y_train, X_test = build_feature_matrix(train, test)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    test = test.copy()
    test['spell'] = model.predict(X_test).astype(int)
    test.to_csv(out_path, index=False)

    return test


# =========================
# 6. Детальная диагностика неизвестных строк
# =========================

def inspect_missing_strings(train: pd.DataFrame, model):
    observed = set(train['rune'])
    missing = sorted(set(all_runes()) - observed)
    if not missing:
        return pd.DataFrame(columns=['rune', 'pred'])

    miss_df = pd.DataFrame({'rune': missing})
    X = build_onehot_matrix(miss_df)
    miss_df['pred'] = model.predict(X).astype(int)
    return miss_df


# =========================
# 7. Запуск всего анализа
# =========================

def full_analysis(train_path='yandex/train_runes.csv', test_path='yandex/test_runes.csv'):
    train, test = load_data(train_path, test_path)

    print('\n=== BASIC CONSISTENCY ===')
    basic = check_basic_consistency(train)

    if len(basic['contradictions']) > 0:
        print('\nЕсть противоречия в train. Дальше анализ бессмысленен, пока они не исправлены.')
        return

    train_feat = build_feature_df(train)

    print('\n=== EXACT HYPOTHESES ===')
    exact_res = test_exact_hypotheses(train_feat)
    print(exact_res.to_string(index=False))

    print('\n=== TRAIN ACCURACY OF SIMPLE MODELS ===')
    model_acc = train_accuracy_of_models(train)
    for k, v in model_acc.items():
        print(f'{k}: {v:.6f}')

    print('\n=== LOO ACCURACY ===')
    loo_res = loo_accuracy(train)
    for k, v in loo_res.items():
        print(f'{k}: {v:.6f}')

    print('\n=== INTERPRETABLE TREE ===')
    tree, depth, rules = fit_interpretable_tree(train)
    print('Minimal depth with perfect train fit:', depth)
    print(rules)

    print('\n=== MISSING STRINGS FROM FULL SPACE ===')
    missing_pred = inspect_missing_strings(train, tree)
    print('Number of unobserved strings:', len(missing_pred))
    if len(missing_pred):
        print(missing_pred.to_string(index=False))

    print('\n=== TEST PREDICTION SAMPLE ===')
    submission = make_submission(train_path, test_path, 'answers.csv')
    print(submission.head())

    print('\nanswers.csv written')


if __name__ == '__main__':
    full_analysis('yandex/train_runes.csv', 'yandex/test_runes.csv')
