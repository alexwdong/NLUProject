for dataset in 20news; do
    for model in ../data/segmentations/$dataset/* ; do
        for threshold in `seq 0.5 0.05 1`; do
            sbatch greene_run_make_cutoff_indices.sbatch $dataset ${model##*/} $threshold
        done
    done
done