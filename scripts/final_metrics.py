"""Ranking metrics and exclusion rules shared by frozen-checkpoint evaluation."""
import math
import numpy as np

METRICS = ('recall_at_20', 'ndcg_at_20', 'tail_share_at_20',
           'coverage_at_20', 'exposure_gini', 'tail_recall_at_20')


def load_interactions(path):
    result = {}
    for line in path.read_text().splitlines():
        values = list(map(int, line.split()))
        if values:
            if values[0] in result:
                raise ValueError(f'Duplicate user row in {path}')
            result[values[0]] = set(values[1:])
    return result


def exclusion_sets(fit, validation, split):
    if split not in ('validation', 'test'):
        raise ValueError('Unknown evaluation split')
    return {u: set(fit.get(u, ())) | (set(validation.get(u, ())) if split == 'test' else set())
            for u in set(fit) | set(validation)}


class RankingMetrics:
    def __init__(self, catalog_size, tail, k=20):
        self.n, self.tail, self.k = catalog_size, set(tail), k
        self.exposure = np.zeros(catalog_size, dtype=np.int64)
        self.recalls, self.ndcgs, self.tail_recalls = [], [], []
        self.tail_slots = 0

    def add(self, recommendations, relevant, seen):
        items = list(map(int, recommendations)); relevant = set(relevant); seen = set(seen)
        if not relevant or not relevant.isdisjoint(seen):
            raise ValueError('Empty or overlapping held-out positives')
        if len(items) != self.k or len(set(items)) != self.k or seen.intersection(items):
            raise ValueError('Invalid or previously observed recommendation')
        if not all(0 <= i < self.n for i in items) or not all(0 <= i < self.n for i in relevant):
            raise ValueError('Item outside the fixed catalog')
        hits = [int(i in relevant) for i in items]
        self.recalls.append(sum(hits) / len(relevant))
        dcg = sum(h / math.log2(j+2) for j,h in enumerate(hits))
        ideal = sum(1 / math.log2(j+2) for j in range(min(self.k,len(relevant))))
        self.ndcgs.append(dcg / ideal)
        np.add.at(self.exposure, items, 1)
        self.tail_slots += sum(i in self.tail for i in items)
        relevant_tail = relevant & self.tail
        if relevant_tail:
            self.tail_recalls.append(len(set(items) & relevant_tail) / len(relevant_tail))

    def finish(self):
        count = len(self.recalls)
        if not count:
            raise ValueError('No eligible evaluation users')
        slots = int(self.exposure.sum()); assert slots == count*self.k
        sorted_counts = np.sort(self.exposure).astype(np.float64)
        return {'evaluation_users':count, 'tail_evaluation_users':len(self.tail_recalls),
                'recommendation_slots':slots, 'catalog_size':self.n,
                'recall_at_20':float(np.mean(self.recalls)),
                'ndcg_at_20':float(np.mean(self.ndcgs)),
                'tail_share_at_20':self.tail_slots/slots,
                'coverage_at_20':float(np.count_nonzero(self.exposure)/self.n),
                'distinct_recommended_items':int(np.count_nonzero(self.exposure)),
                'exposure_gini':float(2*np.dot(np.arange(1,self.n+1),sorted_counts)/(self.n*slots)-(self.n+1)/self.n),
                'tail_recall_at_20':float(np.mean(self.tail_recalls)) if self.tail_recalls else None}
