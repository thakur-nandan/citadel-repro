DATASET=(nfcorpus)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    QRELS_PATH=/store2/scratch/n3thakur/dpr-scale/experiments/datasets/${dataset}/dpr-scale/test.tsv
    TREC_PATH=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/retrieval/retrieval.trec
    python dpr_scale/citadel_scripts/run_beir_eval.py $QRELS_PATH $TREC_PATH > /store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/retrieval/eval_results.txt
done