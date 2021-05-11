for dataset in 20news; do
    for epoch in 500; do
        for model in ../data/embeddings/$dataset/* ; do
            for threshold in 0.95 0.99 0.999 0.9999 0.99999 1.0; do
                sbatch greene_train_LoBERT.sbatch $epoch $model/train/bert_encoded_segments_list_$threshold.pkl
            done
        done
    done
done
