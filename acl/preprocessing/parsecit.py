from lxml import etree
import lxml
import re
import logging
import os

from lxml.etree import LxmlError
from tqdm import tqdm

from acl.utils import normalize_title

logger = logging.getLogger(__name__)


def get_parsecit_files(parscit_dir):
    parscit_files = []

    for d in os.listdir(parscit_dir):
        if os.path.isdir(os.path.join(parscit_dir, d)):  # subdir
            for dd in os.listdir(os.path.join(parscit_dir, d)):  # subdir 2
                if os.path.isdir(os.path.join(parscit_dir, d, dd)):
                    for fn in os.listdir(os.path.join(parscit_dir, d, dd)):  # files
                        fp = os.path.join(parscit_dir, d, dd, fn)

                        parscit_files.append((fn, fp))
    # Total files: 14,714  (server 21,520)
    logger.info(f'Total files: {len(parscit_files):,}')

    return parscit_files


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


def get_citation_pairs_from_parscit(parscit_files, acl_id2s2, title2s2_id):
    # Load citations with s2
    error_files = []
    acl_id2sects = {}
    acl_id2markers = {}
    cit_pairs = []

    # Iterate over papers
    for i, (fn, fp) in enumerate(tqdm(parscit_files, total=len(parscit_files), desc='Reading Parscit files')):
        try:
            sects, cits, markers = load_parscit_file(fp, include_contexts=True)

            from_acl_id = '-'.join(fn.split('-', 2)[:2])  # ACL ID
            acl_id2sects[from_acl_id] = sects
            acl_id2markers[from_acl_id] = markers

            from_s2_id = acl_id2s2[from_acl_id]['paperId'] if from_acl_id in acl_id2s2 else None

            # if from_s2_id not in s2_id2s2_paper:
            #     logger.warning(f'From paper not in index')
            #     continue

            # Citations in paper
            for cit in cits:
                if cit['title'] is None or cit['book_title'] is None or cit['date'] is None:
                    continue

                # Find citing section context
                sect_contexts = []
                for context in cit['contexts']:
                    for i, sect in enumerate(sects):  # Try to find citation string in all sections
                        if context.get('citStr') in sect['text']:
                            # found!
                            sect_contexts.append((sect['generic'], sect['title'], context.get('citStr')))

                # Skip citation if context is not available
                if len(sect_contexts) == 0:
                    continue

                # Find to_s2_id
                cit_title = normalize_title(cit['title'])
                if cit_title in title2s2_id:
                    to_s2_id = title2s2_id[cit_title]

                    for context in sect_contexts:
                        cit_pairs.append(
                            # from_s2_id, (from_acl_id,) to_s2_id, sect_generic, sect_title, sect_marker
                            (
                                from_s2_id,
                                # from_acl_id,
                                to_s2_id,
                            ) + context
                        )
                else:
                    # print('Not found:' + cit_title)
                    pass

        except LxmlError as e:
            error_files.append((fn, fp))
        # if i > 10:
        #    break

    return cit_pairs, error_files
