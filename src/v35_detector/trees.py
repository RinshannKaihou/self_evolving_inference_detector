"""Gradient-boosted trees of a "G" component (calib/tree_select.py), padded to complete binary trees of one depth, so a
boundary's score is `depth` vectorised steps over all trees. One function for the stream (one row) and offline (all rows).

Component spec: {"kind": "G", "feats": [...], "theta": .., "S": .., "trees": {"depth", "base", "feat", "thr", "leaf"}}
  feats  input names: core.FEATS columns (float32, NaN -> -1, as the calibration cache) or 'tpos' (the boundary, raw tokens)
  feat   [n_trees, 2^depth - 1] input index per internal node (heap order: children of k are 2k+1, 2k+2)
  thr    [n_trees, 2^depth - 1] go right iff x > thr (+inf on padding nodes below an early leaf: always left)
  leaf   [n_trees, 2^depth] leaf values (an early leaf's value repeated over the leaves below it)
score = base + sum of the reached leaves; the component holds while score >= theta.
"""
import numpy as np


class Trees:
    def __init__(self, spec):
        self.depth = int(spec['depth'])
        self.feat = np.array(spec['feat'], np.int64)
        self.thr = np.array(spec['thr'], np.float64)
        self.leaf = np.array(spec['leaf'], np.float64)
        self.base = float(spec['base'])
        self._t = np.arange(len(self.feat))

    def score(self, X):
        """X float64 [n, len(feats)] -> score float64 [n]"""
        X = np.asarray(X, np.float64)
        node = np.zeros((len(X), len(self.feat)), np.int64)
        rows = np.arange(len(X))[:, None]
        for _ in range(self.depth):
            node = 2 * node + 1 + (X[rows, self.feat[self._t, node]] > self.thr[self._t, node])
        return self.base + self.leaf[self._t, node - (2 ** self.depth - 1)].sum(1)


def inputs(vals, t):
    """one input row from feature values (float32 semantics, NaN -> -1) and 'tpos' entries given as None"""
    pos = [i for i, v in enumerate(vals) if v is None]
    x = np.array([np.nan if v is None else v for v in vals], np.float64).astype(np.float32).astype(np.float64)
    x[np.isnan(x)] = -1.0
    x[pos] = float(t)
    return x


def export(model, depth):
    """sklearn HistGradientBoostingClassifier (binary, numeric features, no NaN inputs) -> spec dict (complete trees)"""
    n_int, n_leaf = 2 ** depth - 1, 2 ** depth
    feat, thr, leaf = [], [], []
    for (pred,) in model._predictors:
        nodes = pred.nodes
        f, h, v = np.zeros(n_int, np.int64), np.full(n_int, np.inf), np.zeros(n_leaf)

        def walk(i, k, d):                              # sklearn node i placed at heap slot k, depth d
            if nodes['is_leaf'][i]:
                span = 2 ** (depth - d)                 # leaves under heap slot k at the bottom level
                first = (k + 1) * span - 1 - n_int
                v[first:first + span] = nodes['value'][i]
                return                                  # internal slots below keep feat 0 / thr +inf: always left
            if d >= depth:
                raise ValueError('tree deeper than depth')
            f[k], h[k] = nodes['feature_idx'][i], nodes['num_threshold'][i]
            walk(nodes['left'][i], 2 * k + 1, d + 1)
            walk(nodes['right'][i], 2 * k + 2, d + 1)

        walk(0, 0, 0)
        feat.append(f.tolist())
        thr.append(h.tolist())
        leaf.append(v.tolist())
    return {'depth': depth, 'base': float(np.ravel(model._baseline_prediction)[0]), 'feat': feat, 'thr': thr, 'leaf': leaf}
