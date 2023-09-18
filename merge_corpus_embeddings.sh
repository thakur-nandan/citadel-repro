
export CUDA_VISIBLE_DEVICES=3
DATASET=(nfcorpus)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    OUTPUT_DIR=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/merged_embeddings/
    CTX_EMBEDDINGS_DIR=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/corpus_embeddings/
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/merge_experts.py $OUTPUT_DIR "$CTX_EMBEDDINGS_DIR" "0-31000"
done