# This script merges the embeddings of the corpus for the DPR model.

export CUDA_VISIBLE_DEVICES=0
DATASET=(webis-touche2020-v3)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    OUTPUT_DIR=output/${dataset}/merged_embeddings/
    CTX_EMBEDDINGS_DIR=output/${dataset}/corpus_embeddings/
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/merge_experts.py $OUTPUT_DIR "$CTX_EMBEDDINGS_DIR" "0-31000"
done