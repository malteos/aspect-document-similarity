import logging

import bibtexparser
from lxml import etree
import lxml
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import os
import pickle
import time
import json
import re
import numpy as np
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
import requests
from lxml.etree import  LxmlError

logger = logging.getLogger(__name__)


def load_acl_corpus(data_dir):
    title2acl_ids = defaultdict(list)
    acl_id2meta = {}

    year2titles = defaultdict(list)
    author_last2titles = defaultdict(list)

    parser = etree.XMLParser(recover=True)

    for d in os.listdir(os.path.join(data_dir, 'aclxml')):
        if os.path.isdir(os.path.join(data_dir, 'aclxml', d)):
            for vol in os.listdir(os.path.join(data_dir, 'aclxml', d)):
                if os.path.isdir(os.path.join(data_dir, 'aclxml', d, vol)):
                    xml_fp = os.path.join(data_dir, 'aclxml', d, vol, vol + '.xml')
                    # print(vol)

                    tree = etree.parse(xml_fp, parser=parser)

                    # Parse volume
                    papers = tree.getroot().cssselect('paper')

                    for paper in papers:
                        title = next(iter(paper.xpath('./title/text()')), None)
                        year = next(iter(paper.xpath('./year/text()')), None)
                        authors_first = paper.xpath('./author/first/text()')
                        authors_last = paper.xpath('./author/last/text()')

                        if title is None or year is None:
                            continue

                        acl_id = vol + '-' + paper.get('id')

                        acl_id2meta[acl_id] = dict(
                            title=title,
                            year=year,
                            book_title=next(iter(paper.xpath('./booktitle/text()')), None),
                            bibkey=next(iter(paper.xpath('./bibkey/text()')), None),
                            authors_first=authors_first,
                            authors_last=authors_last,

                        )
                        title2acl_ids[title].append(acl_id)
                        year2titles[year].append(title)

                        for last in authors_last:
                            author_last2titles[last].append(title)

    # Extracted titles: 14,760
    print(f'Extracted titles: {len(title2acl_ids):,}')

    return title2acl_ids, acl_id2meta, year2titles, author_last2titles


def get_text_with_cssselect(ele, selector, default=None, ith=0):
    s = ele.cssselect(selector)

    if len(s) > ith:
        return s[ith].text
    else:
        return default


def load_parscit_file(fp, include_contexts=False):
    # read from file path
    tree = etree.parse(fp)

    # sections
    algo_sect = tree.getroot().cssselect('algorithm[name="SectLabel"] > variant')[0]
    sects = []
    sect = None

    for child in algo_sect.getchildren():
        if child.tag == 'sectionHeader':
            sects.append({
                'title': child.text.strip(),
                'generic': child.get('genericHeader'),
                'text': '',
            })

        elif child.tag == 'bodyText':
            # Create untitled section if none exist
            if len(sects) == 0:
                sects.append({
                    'title': None,
                    'generic': None,
                    'text': '',
                })

            # Append to last section
            sects[-1]['text'] += child.text.strip()

    # replace line breaks within sentence (could be improved)
    for i, sect in enumerate(sects):
        sects[i]['text'] = re.sub(r'([A-Za-z],;)([\r\n]+)([A-Za-z])', r'\1 \3', sect['text'])

    # Iterate over all valid citations
    cits = []

    def get_text_with_cssselect(ele, selector, default=None, ith=0):
        s = ele.cssselect(selector)

        if len(s) > ith:
            return s[ith].text
        else:
            return default

    for cit_ele in tree.getroot().cssselect('algorithm[name="ParsCit"] > citationList > citation[valid="true"]'):
        try:

            title = get_text_with_cssselect(cit_ele, 'title')
            marker = get_text_with_cssselect(cit_ele, 'marker')
            date = get_text_with_cssselect(cit_ele, 'date')  # str
            book_title = get_text_with_cssselect(cit_ele, 'booktitle')

            authors = [e.text for e in cit_ele.cssselect('authors > author')]

            if date and len(date) != 4:
                raise ValueError(f'Invalid date: {date}')
            cit = dict(title=title, authors=authors, marker=marker, date=date, book_title=book_title)

            if include_contexts:
                cit['contexts'] = cit_ele.cssselect('contexts > context')

            cits.append(cit)
        except IndexError as e:
            print(f'Cannot parse citation: {e}; {etree.tostring(cit_ele)[:100]}')

    # Extract all citation markers (for later cleaning from section text)
    markers = []
    for cit_context in tree.getroot().cssselect(
            'algorithm[name="ParsCit"] > citationList > citation > contexts > context'):
        if 'citStr' in cit_context.attrib:
            markers.append(cit_context.get('citStr'))

    return sects, cits, markers


# Extract citation context
# - find section in which the citation markers can be found
# - find the corresponding ACL paper
# - fuzzy title search is expensive, therefore, we check on year + authors first to decrease search space.
def get_citation_context(cits, sects, title2acl_ids, year2titles, author_last2titles):
    cits_with_context = []  # (bib_idx, sect_context)

    for cit in cits:
        if cit['title'] is None or cit['book_title'] is None or cit['date'] is None:
            continue

        # Find section context
        sect_contexts = []
        for context in cit['contexts']:
            for i, sect in enumerate(sects):  # Try to find citation string in all sections
                if context.get('citStr') in sect['text']:
                    # found!
                    # print(sect['title'])
                    # print(sect['generic'])
                    sect_contexts.append((sect['generic'], sect['title'], context.get('citStr')))

            # print(context.get('citStr'))
            # print(context.get('position'))
            # print(context.get('startWordPos'))

        if len(sect_contexts) == 0:
            continue

        # Filter for ACL proceedings
        # TODO could be improved
        if 'ACL' in cit['book_title'] or 'Linguistics' in cit['book_title']:
            year_candidates = set(year2titles[cit['date']])  # papers from the same year

            if len(year_candidates) > 0:
                # papers from authors with same name
                # note: all name parts are used, bc we do not know what the first or last name is.
                author_names = [name for author in cit['authors'] for name in author.split()]
                author_candidates = []
                for name in author_names:
                    if name in author_last2titles:
                        author_candidates += author_last2titles[name]
                author_candidates = set(author_candidates)

                if len(author_candidates) > 0:
                    # candidate must be in both sets
                    candidates = year_candidates & author_candidates

                    if len(candidates) > 0:
                        match_title, score = process.extractOne(cit['title'], candidates)

                        # Candidate must be above threshold
                        if score > .95 and match_title in title2acl_ids:
                            for acl_id in title2acl_ids[match_title]:
                                # Citation found in bib
                                for sc in sect_contexts:
                                    cits_with_context.append((acl_id, sc))

                # bib_candidates = process.extract(cit['title'], candidate_titles, limit=1)
                # for c_title, score in bib_candidates:
                #    for acl_id in title2acl_ids[c_title]:
                #        # Citation found in bib
                #        for sc in sect_contexts:
                #            cits_with_context.append((acl_id, sc))

                # TODO multi title matches? -> check for year

                # print(c_idx)
                # print(bib_database.entries[c_idx]['title'])
                # print(marker)
                #    break
    return cits_with_context


