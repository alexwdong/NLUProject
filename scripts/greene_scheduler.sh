for dataset in 20news; do
    for model in ../data/segmentations/$dataset/* ; do
#         for threshold in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.96 0.97 0.98 0.99 0.999 0.9999 0.99999 0.999999 1; do
          for threshold in 0.95 0.99 0.999 0.9999 0.99999 1; do
            sbatch -J "${1%.*}" greene_run_script.sbatch $1 $dataset ${model##*/} $threshold
        done
    done
done
