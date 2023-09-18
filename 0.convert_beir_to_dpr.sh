DATASET=(webis-touche2020)

for dataset in ${DATASET[*]}
do
    echo $dataset
    # mkdir -p /store2/scratch/n3thakur/dpr-scale/experiments/data/$dataset
    output_path=/store2/scratch/n3thakur/dpr-scale/experiments/data/$dataset
    dataset_path=/store2/scratch/n3thakur/touche-ablations/beir-datasets-touche-d2q/
    python /store2/scratch/n3thakur/dpr-scale/dpr_scale/citadel_scripts/convert_beir_to_dpr_format.py $dataset_path $output_path
done
