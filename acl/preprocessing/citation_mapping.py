import json
import re
from typing import List


def get_title2s2_id(id2s2__title_list: List):
    title2s2_id = {}

    for id2s2, id2title in id2s2__title_list:
        title2s2_id.update({id2title[_id]: s2['paperId'] for _id, s2 in id2s2.items() if _id in id2title})

    return title2s2_id


def get_dblp_titles(fp):
    """


    :param fp: Path to DBLP scraper results (JSON)
    :return: acl_id2title, doi2title, arxiv2title
    """
    title2dblp_hits = json.load(open(fp, 'r'))

    title2doi = {}
    doi2title = {}

    title2arxiv = {}
    arxiv2title = {}

    title2acl_id = {}
    acl_id2title = {}

    for i, (title, hits) in enumerate(title2dblp_hits.items()):
        if hits['@total'] == '1':  # igore multi matches
            hit = hits['hit'][0]

            if 'doi' in hit['info']:
                doi = hit['info']['doi'].replace('https://doi.org/', '')

                doi2title[doi] = title
                title2doi[title] = doi
                continue

            if 'ee' in hit['info']:
                ee = hit['info']['ee']
                if 'aclweb.org/anthology/' in ee:
                    match = re.search(r'anthology/([-a-zA-Z0-9]+)', ee)
                    if match:
                        acl_id = match.group(1)
                        title2acl_id[title] = acl_id
                        acl_id2title[acl_id] = title
                        continue

                    # print(acl_id)

                if 'arxiv.org' in ee:
                    match = re.search(r'arxiv.org\/abs\/(.+)', ee)
                    if match:
                        arxiv_id = match.group(1)
                        title2arxiv[title] = arxiv_id
                        arxiv2title[arxiv_id] = title
                        continue

                    # print(arxiv_id)
                # other
                # print(hit['info']['ee'])

        #    print(hits)
        #    print('----')
        #    if i > 100:
        #        break

    found = len(doi2title) + len(arxiv2title) + len(acl_id2title)

    print(f'Found DOIs: {len(doi2title)} ({len(title2doi)})')
    print(f'Found arXiv: {len(arxiv2title)}')
    print(f'Found ACL: {len(acl_id2title)}')

    print(f'-- Found all: {found:,} / {len(title2dblp_hits):,}')

    return acl_id2title, doi2title, arxiv2title


def get_s2_pairs_from_cits(cit_pairs, acl_id2s2):
    s2_pairs = []
    not_found = []

    for from_s2_id, from_acl_id, to_s2_id, sect_generic, sect_title, sect_marker in cit_pairs:
        if from_s2_id == None:
            if from_acl_id in acl_id2s2:
                from_s2_id = acl_id2s2[from_acl_id]['paperId']
            else:
                not_found.append((from_acl_id, to_s2_id))
                continue

        s2_pairs.append((
            from_s2_id,
            to_s2_id,
            sect_generic,
            sect_title,
            sect_marker,
        ), )

    return s2_pairs, not_found
