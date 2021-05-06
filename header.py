import os

# Run the following to set up a symbolic link
# ln -s /scratch/awd275/NLU_data ~/NLU_data

# Run `exec(open("../header.py").read())` at the beginning of each script to use header

DATA_DIR = lambda x = '':os.path.expanduser(f'~/NLU_data/{x}')
RAW_DIR = lambda x = '':DATA_DIR(f'raw/{x}')
SEGMENT_DIR = lambda x = '':DATA_DIR(f'segmentations/{x}')
EMBEDDINGS_DIR = lambda x = '':DATA_DIR(f'embeddings/{x}')
RESULTS_DIR = lambda x = '':DATA_DIR(f'results/{x}')

all_roots = {
    'DATA_DIR':DATA_DIR(),
    'RAW_DIR':RAW_DIR(),
    'SEGMENT_DIR':SEGMENT_DIR(),
    'EMBEDDINGS_DIR':EMBEDDINGS_DIR(),
    'RESULTS_DIR':RESULTS_DIR()
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