import os

data_dir = lambda x:os.path.expanduser(f'~/NLU_data/{x}')
raw_dir = 

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