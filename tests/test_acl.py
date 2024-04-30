import os
from collections import defaultdict
from unittest import TestCase

from acl.__data_prep import load_parscit_file, get_citation_context
from experiments.environment import get_env


class ACLTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = get_env()
        self.data_dir = data_dir = os.path.join(self.env['datasets_dir'], 'acl-anthology')

    def test_get_cits(self):
        title2acl_ids = {}
        author_last2titles = {}
        year2titles = defaultdict(list)

        fp = self.data_dir + '/parscit/D/D15/D15-1312-parscit.130908.xml'
        fn = 'D15-1312-parscit.130908.xml'

        error_files = []
        out = []
        acl_id2sects = {}
        acl_id2markers = {}

        sects, cits, markers = load_parscit_file(fp)

        from_id = '-'.join(fn.split('-', 2)[:2])  # ACL ID

        acl_id2sects[from_id] = sects
        acl_id2markers[from_id] = markers

        print('----')

        print(sects)

        print('----')

        print(cits)

        print('---')

        cits_with_context = get_citation_context(cits, sects, title2acl_ids, year2titles, author_last2titles)
        #
        # out += [(from_id, to_id, context[0], context[1], context[2]) for to_id, context in cits_with_context]

