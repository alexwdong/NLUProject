for dataset in 20news; do
    for model in ../data/segmentations/$dataset/* ; do
        for threshold in `seq 0.5 0.05 1`; do
            sbatch -J "${1%.*}" greene_run_script.sbatch $1 $dataset ${model##*/} $threshold
        done
    done
done