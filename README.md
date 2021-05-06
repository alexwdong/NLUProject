# NLUProject
Text Segmentation to Improve Long Document Understanding
 
# Outline:

scripts folder contains the majority of the work done. There are three main sets of scripts

1) Make_baseline_bert_encoded_segments.py
    This makes BERT encoded segments in the overlap scheme described in Pappagari 2019 (e.g each segment is 200 tokens long, and there is a 50 token shift between two consecutive segments, which creates a 150 token overlap).

2) These scripts are meant to be run in succession. Together these create BERT encoded segments in our initial proposed idea (using BERT's NSP prediction task to create segments before encoding).
2a) make_text_splits_dict.py
    This script makes "qid_struct" files (which stand for question_id structures, question not really the right term for the 20newsgroup dataset). The qid_struct files contain the NSP sequences for each long document. For example, a long document might be composed of 100 sentences. the NSP sequence will contain 99 probabilities, for determining a split between each consecutive pair of sentences. For the 20 newsgroup dataset, the way that the dataset was structured made it conducive to make 20 qid structures, one for each of the 20 labels. 
2b) make_cutoff_indices.py
    This script takes the "qid_struct" files and a probability threshold (to create NSP splits), and creates a label_to_cutoff_indices_dict pickle file, which for 20 newsgroups, is structured as follows. It is a dictionary of 20 keys (representing the labels) which maps to another dictionary, which maps the index of the long document (for the just mentioned label) to a sequence of cutoff indices. The cutoff indices indicate which NSP probabilities were below the threshold. For example. The label "atheism" has 480 long documents associated with it, so label_to_cutoff_indices_dict.keys() = [0,1,...,479]. The first document (with index 0) has 109 sentences, which might be split at indices [2,4,..,98,100], and so label_to_cutoff_indices_dict['atheism'][0] = [2,4,...,98,100]
    
2c) make_bert_encoded_segments.py
This takes a label_to_cutoff_indices_dict pickle file and creates BERT encoded segments. The label_to_cutoff_indices contains the split indices for each long document. This function applies those splits to each document, which creates a series of "segments", then uses BERT to encode those segments, and saves it into bert_encoded_segments_list pickle, which is a list of 2-tuples of (label, bert_encoded_segment). We use BERT to encode each segment as follows. We take the segment, tokenize it (appending the CLS token), and then take the last hidden state of the CLS token. Note that a paper (SentenceBERT) showed that using this method to encode sentences seemed to not work too well, that being said, we wanted to follow the methodology of Pappagari for comparison.

3) train_LoBERT.py 
    This trains an LSTM over BERT model (as defined in Pappagari), taking as input a bert_encoded_segmentns file. It creates two outputs, a results pkl file with train/val loss/accuracy, and a best_model file, which was LoBERT model with the best validation accuracy over training.
    












