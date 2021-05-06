import os

# Run the following to set up a symbolic link
# ln -s /scratch/awd275/NLU_data ~/NLU_data

DATA_DIR = lambda x = '':os.path.expanduser(f'~/NLU_data/{x}')
RAW_DIR = lambda x = '':data_dir(f'raw/{x}')
SEGMENT_DIR = lambda x = '':data_dir(f'segmentations/{x}')
EMBEDDINGS_DIR = lambda x = '':data_dir(f'embeddings/{x}')
RESULTS_DIR = lambda x = '':data_dir(f'results/{x}')

all_roots = {
    'data_dir':DATA_DIR(),
    'raw_dir':RAW_DIR(),
    'segment_dir':SEGMENT_DIR(),
    'embeddings_dir':EMBEDDINGS_DIR(),
    'results_dir':RESULTS_DIR()
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