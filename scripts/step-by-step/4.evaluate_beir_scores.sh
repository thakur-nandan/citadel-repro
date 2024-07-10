# Evaluate the CITADEL+ retrieved results on the BEIR dataset.

DATASET=(webis-touche2020-v3)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    QRELS_PATH=datasets/${dataset}/dpr-scale/test.tsv
    TREC_PATH=output/${dataset}/retrieval/retrieval.trec
    python dpr_scale/citadel_scripts/run_beir_eval.py $QRELS_PATH $TREC_PATH > output/${dataset}/retrieval/eval_results.txt
done