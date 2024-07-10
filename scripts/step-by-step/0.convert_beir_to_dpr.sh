# This code is used to convert BEIR dataset into DPR format used for CITADEL+ experiments.
# Download the `webis-touche2020-v3` dataset and keep it under the `data` directory as `webis-touche2020-v3-beir`.

DATASET=(webis-touche2020-v3)

for dataset in ${DATASET[*]}
do
    echo $dataset
    mkdir -p data/$dataset
    output_path=data/$dataset
    dataset_path=data/$dataset-beir
    python dpr_scale/citadel_scripts/convert_beir_to_dpr_format.py $dataset_path $output_path
done
