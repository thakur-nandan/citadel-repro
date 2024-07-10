# Generate the corpus embeddings for the BEIR dataset using the CITADEL+ model.

export CUDA_VISIBLE_DEVICES=0

# Download the checkpoint
# check if the checkpoint exists
mkdir -p checkpoints/
wget https://dl.fbaipublicfiles.com/citadel/checkpoints/citadel/citadel_plus/checkpoint_best.ckpt -P checkpoints/

# Set the checkpoint path
CHECKPOINT_PATH=checkpoints/checkpoint_best.ckpt

DATASET=(webis-touche2020-v3)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    CTX_EMBEDDINGS_DIR=output/${dataset}/corpus_embeddings/
    DATA_PATH=datasets/${dataset}/dpr-scale/corpus.tsv

    HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH nohup python dpr_scale/citadel_scripts/generate_multivec_embeddings.py -m --config-name msmarco_aws.yaml \
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
    trainer=gpu_1_host trainer.gpus=1 > nohup_${dataset}.log 2>&1&
done