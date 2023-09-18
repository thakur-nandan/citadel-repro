# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import pytrec_eval
import logging
from typing import List, Dict, Tuple
import sys
import csv
import collections

def hole(qrels: Dict[str, Dict[str, int]], 
               results: Dict[str, Dict[str, float]], 
               k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    Hole = {}
    
    for k in k_values:
        Hole[f"Hole@{k}"] = 0.0
    
    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():    
            annotated_corpus.add(doc_id)
    
    k_max = max(k_values)
    print("\n")
    
    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            Hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"]/len(qrels), 5)
        print("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))


def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        if ignore_identical_ids:
            logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
        for metric in [ndcg, _map, recall, precision]:
            print("\n")
            for k in metric.keys():
                print("{}: {:.4f}".format(k, metric[k]))

def load_qrels(path):
    qrels = {}
    reader = csv.reader(open(path, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    for row in reader:
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: int(score)}
        else:
            qrels[query_id][corpus_id] = int(score)
    return qrels

def load_results(path):
    with open(path) as f:
        lines = f.readlines()
    results = collections.defaultdict(dict)
    for line in lines:
        qid, _, doc_id, _, score, _ = line.strip().split(" ")
        results[qid][doc_id] = float(score)
    return results

def main():
    qrels = load_qrels(sys.argv[1])
    results = load_results(sys.argv[2])
    evaluate(qrels, results, [1, 3, 5, 10, 100, 1000])
    hole(qrels, results, [1, 3, 5, 10, 100, 1000])

if __name__ == "__main__":
    main()
