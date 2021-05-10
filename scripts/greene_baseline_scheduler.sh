for dataset in 20news; do
    for model in ../data/embeddings/$dataset/* ; do
        for sequence_length in `seq 100 50 300`; do
            for shift_length in 1 2 4 8; do
               sbatch greene_run_baseline.sbatch $dataset ${model##*/} $sequence_length $(($sequence_length/$shift_length))
             done
         done
    done
done
