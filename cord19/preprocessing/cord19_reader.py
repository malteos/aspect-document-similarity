import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


def get_dict_value(d, key, default=None):
    if key in d:
        return d[key]
    else:
        return default


def get_papers_and_citations_from_cord19(input_dir, id2meta):

    subsets = ['biorxiv_medrxiv', 'comm_use_subset', 'custom_license', 'noncomm_use_subset']
    id2paper = {}

    has_doi = 0
    bib_count = 0
    cits = []  # from_doi, to_doi, <section title>

    for ss in subsets:
        ss_dir = os.path.join(input_dir, ss)

        # iterate over files
        for fn in os.listdir(ss_dir):
            if not fn.endswith('.json'):
                continue

            fp = os.path.join(ss_dir, fn)
            with open(fp, 'r') as f:
                paper = json.load(f)

                if paper['paper_id'] not in id2meta:
                    continue

                meta = id2meta[paper['paper_id']]

                paper['_meta'] = dict(meta)

                id2paper[paper['paper_id']] = paper

                # has valid DOI
                if isinstance(meta['doi'], str) and len(meta['doi']) > 10:
                    # iterate over body text
                    for paragraph in paper['body_text']:
                        # iterate over each citation marker
                        for cit in paragraph['cite_spans']:
                            # find corresponding bib entry
                            if cit['ref_id'] in paper['bib_entries']:
                                bib = paper['bib_entries'][cit['ref_id']]
                                bib_count += 1

                                # only use bib entries with DOI
                                if 'DOI' in bib['other_ids']:
                                    has_doi += 1

                                    for out_doi in bib['other_ids']['DOI']:
                                        cits.append((
                                            meta['doi'],
                                            out_doi,
                                            paragraph['section']
                                        ))
        # break
    # break

    logger.info(f'Paper count: {len(id2paper)}')
    logger.info(f'DOI exists: {has_doi / bib_count} (total: {bib_count}; doi: {has_doi})')
    logger.info(f'Citation pairs: {len(cits)}')

    return id2paper, cits


def merge_cord19_and_s2_papers(id2paper, id2meta, doi2s2paper: Dict[str, Dict]) -> Dict[str, Dict]:
    """

    Merge CORD-19 + S2

    :param id2meta:
    :param id2paper:
    :param doi2s2paper:
    :return: DOI => Paper
    """
    doi2paper = {}

    for pid, cord_paper in id2paper.items():
        if pid in id2meta:
            doi = id2meta[pid]['doi']

            paper = {
                'cord19_id': cord_paper['paper_id'],
                's2_id': None,
                'title': cord_paper['metadata']['title'],
                'abstract': cord_paper['abstract'][0]['text'] if len(cord_paper['abstract']) == 1 else None,
                'arxivId': None,
                'doi': doi,
                'venue': cord_paper['_meta']['journal'],
                'year': int(cord_paper['_meta']['publish_time'].split('-')[0]),
                'citations_count': None,
                'references_count': len(cord_paper['bib_entries']),
                'authors': [author['first'] + ' ' + author['last'] for author in cord_paper['metadata']['authors']],
            }
            doi2paper[doi] = paper

    for doi, s2 in doi2s2paper.items():

        paper = {
            'cord19_id': None,
            's2_id': get_dict_value(s2, 'paperId'),
            'title': get_dict_value(s2, 'title'),
            'abstract': get_dict_value(s2, 'abstract'),
            'doi': doi,
            'arxivId': get_dict_value(s2, 'arxivId'),
            'venue': get_dict_value(s2, 'venue'),
            'year': get_dict_value(s2, 'year', 0),
            'citations_count': len(get_dict_value(s2, 'citations', [])),
            'references_count': len(get_dict_value(s2, 'references', [])),
            'authors': [a['name'] for a in get_dict_value(s2, 'authors', []) if 'name' in a],
        }

        if doi in doi2paper:
            logger.warning(f'Overriding CORD19 with S2 paper data: {doi}')

            paper['cord19_id'] = doi2paper[doi]['cord19_id']

        doi2paper[doi] = paper

    return doi2paper
