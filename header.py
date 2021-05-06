import os

# Run the following to set up a symbolic link
# ln -s /scratch/awd275/NLU_data ~/NLU_data

data_dir = lambda x = '':os.path.expanduser(f'~/NLU_data/{x}')
raw_dir = lambda x = '':data_dir(f'raw/{x}')
segment_dir = lambda x = '':data_dir(f'segmentations/{x}')
embeddings_dir = lambda x = '':data_dir(f'embeddings/{x}')
results_dir = lambda x = '':data_dir(f'results/{x}')

all_roots = {
    'data_dir':data_root(),
    'raw_dir':raw_root(),
    'segment_dir':segment_root(),
    'embeddings_dir':embeddings_root(),
    'results_dir':results_root()
}

newsgroup_configs = ['bydate_alt.atheism',
                     'bydate_comp.graphics',
                     'bydate_comp.os.ms-windows.misc',
                     'bydate_comp.sys.ibm.pc.hardware',
                     'bydate_comp.sys.mac.hardware',
                     'bydate_comp.windows.x',
                     'bydate_misc.forsale',
                     'bydate_rec.autos',
                     'bydate_rec.motorcycles',
                     'bydate_rec.sport.baseball',
                     'bydate_rec.sport.hockey',
                     'bydate_sci.crypt',
                     'bydate_sci.electronics',
                     'bydate_sci.med',
                     'bydate_sci.space',
                     'bydate_soc.religion.christian',
                     'bydate_talk.politics.guns',
                     'bydate_talk.politics.mideast',
                     'bydate_talk.politics.misc',
                     'bydate_talk.religion.misc']