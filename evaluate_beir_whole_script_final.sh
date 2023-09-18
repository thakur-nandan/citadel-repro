export CUDA_VISIBLE_DEVICES=4,5,6,7
DATASET=(webis-touche2020-fff-touche-d2q-wo-title-original-annotations)

# Model Checkpoint path
CHECKPOINT_PATH=/store2/scratch/n3thakur/dpr-scale/experiments/checkpoints/citadel_plus_checkpoint_best.ckpt

# Data directory
DATA_DIR=/store2/scratch/n3thakur/touche-ablations/beir-datasets-touche-d2q

# BEIR Dataset paths
for dataset in ${DATASET[*]}
do
    I2D_PATH=$DATA_DIR/dpr-scale/index2docid.tsv
    DATA_PATH=$DATA_DIR/dpr-scale/corpus.tsv
    PATH_TO_QUERIES_TSV=$DATA_DIR/dpr-scale/queries.tsv
done

# Generate Corpus Embeddings for the BEIR Dataset
for dataset in ${DATASET[*]} 
do
    echo $dataset
    CTX_EMBEDDINGS_DIR=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/corpus_embeddings/
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


# Merge corpus embeddings for the BEIR Dataset
for dataset in ${DATASET[*]} 
do
    echo $dataset
    OUTPUT_DIR=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/merged_embeddings/
    CTX_EMBEDDINGS_DIR=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/corpus_embeddings/
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/merge_experts.py $OUTPUT_DIR "$CTX_EMBEDDINGS_DIR" "0-31000"
done

# Evaluate the BEIR Dataset using the merged embeddings
for dataset in ${DATASET[*]}
do
    OUTPUT_DIR=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/retrieval
    MERGED_EMBEDDINGS_DIR=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/merged_embeddings/expert
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


# Evaluate the BEIR Dataset using the merged embeddings -- get scores from TREC file
for dataset in ${DATASET[*]} 
do
    echo $dataset
    QRELS_PATH=$DATA_DIR/dpr-scale/test.tsv
    TREC_PATH=/store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/retrieval/retrieval.trec
    python dpr_scale/citadel_scripts/run_beir_eval.py $QRELS_PATH $TREC_PATH > /store2/scratch/n3thakur/dpr-scale/experiments/output/${dataset}/retrieval/eval_results.txt
done