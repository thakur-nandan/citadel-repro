# This script encodes the query embeddings and does the search using the corpus embeddings.

export CUDA_VISIBLE_DEVICES=0

export dataset=webis-touche2020-v3
OUTPUT_DIR=output/${dataset}/retrieval
CTX_EMBEDDINGS_DIR=output/${dataset}/merged_embeddings/expert
CHECKPOINT_PATH=checkpoints/checkpoint_best.ckpt

I2D_PATH=datasets/${dataset}/dpr-scale/index2docid.tsv
DATA_PATH=datasets/${dataset}/dpr-scale/corpus.tsv
PATH_TO_QUERIES_TSV=datasets/${dataset}/dpr-scale/queries.tsv

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
+task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
+task.checkpoint_path=$CHECKPOINT_PATH \
+task.index2docid_path=$I2D_PATH \
+task.passages=$DATA_PATH \
+task.portion="$PORTION" \
+task.topk=1000 +task.cuda=True +task.quantizer=None +task.sub_vec_dim=4 trainer.precision=16 +task.expert_parallel=True \
trainer=gpu_1_host trainer.gpus=1
