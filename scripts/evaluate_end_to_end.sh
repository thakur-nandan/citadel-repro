# Description: Evaluate the end-to-end CITADEL+ pipeline for the BEIR Dataset
# Can be used to encode and search using multiple GPUs.
# Usage: bash evaluate_end_to_end.sh
# Requirements: CITADEL+ model checkpoint, BEIR Dataset, and the CITADEL+ model codebase.

export CUDA_VISIBLE_DEVICES=4,5,6,7

# BEIR Dataset
DATASET=(webis-touche2020-v3)

# Model Checkpoint path
CHECKPOINT_PATH=checkpoints/citadel_plus_checkpoint_best.ckpt

# Data directory
DATA_DIR=data/$DATASET

#### STEP 0: Convert BEIR Dataset to DPR format ####
for dataset in ${DATASET[*]}
do
    echo $dataset
    mkdir -p data/$dataset
    output_path=data/$dataset
    dataset_path=data/$dataset-beir
    python dpr_scale/citadel_scripts/convert_beir_to_dpr_format.py $dataset_path $output_path
done

# BEIR Dataset paths
for dataset in ${DATASET[*]}
do
    I2D_PATH=$DATA_DIR/dpr-scale/index2docid.tsv
    DATA_PATH=$DATA_DIR/dpr-scale/corpus.tsv
    PATH_TO_QUERIES_TSV=$DATA_DIR/dpr-scale/queries.tsv
done

#### Step 1: Generate Corpus Embeddings for the BEIR Dataset ####
for dataset in ${DATASET[*]} 
do
    echo $dataset
    CTX_EMBEDDINGS_DIR=output/${dataset}/corpus_embeddings/
    DATA_PATH=$DATA_DIR/dpr-scale/corpus.tsv

    HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/generate_multivec_embeddings.py -m --config-name msmarco_aws.yaml \
    datamodule=generate \
    datamodule.test_path=$DATA_PATH \
    task=multivec task/model=citadel_model \
    task.model.tok_projection_dim=32 task.model.cls_projection_dim=128 \
    task.shared_model=True \
    +task.add_cls=True \
    +task.query_topk=1 +task.context_topk=5 \
    +task.weight_threshold=0.0 \
    +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
    +task.checkpoint_path=$CHECKPOINT_PATH \
    +task.add_context_id=False \
    trainer=gpu_1_host trainer.gpus=4
done


#### Step 2: Merge corpus embeddings for the BEIR Dataset ####
for dataset in ${DATASET[*]} 
do
    echo $dataset
    OUTPUT_DIR=output/${dataset}/merged_embeddings/
    CTX_EMBEDDINGS_DIR=output/${dataset}/corpus_embeddings/
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/merge_experts.py $OUTPUT_DIR "$CTX_EMBEDDINGS_DIR" "0-31000"
done

# Step 3: Evaluate the BEIR Dataset using the merged embeddings ####
for dataset in ${DATASET[*]}
do
    OUTPUT_DIR=output/${dataset}/retrieval
    MERGED_EMBEDDINGS_DIR=output/${dataset}/merged_embeddings/expert
done

PORTION=1.0  #0.001 # how much portion of the index should be moved to GPU before retrieval

HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_citadel_retrieval.py --config-name msmarco_aws.yaml \
datamodule=generate_multivec_query_emb \
datamodule.test_path=$PATH_TO_QUERIES_TSV \
datamodule.test_batch_size=1 \
+datamodule.trec_format=True \
task=multivec_retrieval task/model=citadel_model \
task.model.tok_projection_dim=32 \
task.model.cls_projection_dim=128 +task.add_cls=True task.shared_model=True \
+task.query_topk=1 +task.context_topk=5 \
+task.output_path=$OUTPUT_DIR \
+task.ctx_embeddings_dir=$MERGED_EMBEDDINGS_DIR \
+task.checkpoint_path=$CHECKPOINT_PATH \
+task.index2docid_path=$I2D_PATH \
+task.passages=$DATA_PATH \
+task.portion="$PORTION" \
+task.topk=1000 +task.cuda=True +task.quantizer=None +task.sub_vec_dim=4 trainer.precision=16 +task.expert_parallel=True \
trainer=gpu_1_host trainer.gpus=4


#### Step 4: Evaluate the BEIR Dataset using the merged embeddings -- get scores from TREC file ####
for dataset in ${DATASET[*]} 
do
    echo $dataset
    QRELS_PATH=$DATA_DIR/dpr-scale/test.tsv
    TREC_PATH=output/${dataset}/retrieval/retrieval.trec
    python dpr_scale/citadel_scripts/run_beir_eval.py $QRELS_PATH $TREC_PATH > output/${dataset}/retrieval/eval_results.txt
done