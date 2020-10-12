# shared author
import logging
import math
import random
from collections import defaultdict
from typing import List, Tuple, Set

from fuzzywuzzy import fuzz

from acl.utils import get_sorted_pair

logger = logging.getLogger(__name__)


def get_cocitations(cits: List[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    from_to_cits = defaultdict(set)

    for from_id, to_id in cits:
        from_to_cits[from_id].add(to_id)

    cocits_set = set()

    for from_cit, to_cits in from_to_cits.items():
        for a in to_cits:
            for b in to_cits:
                cocits_set.add(get_sorted_pair(a, b))

    logger.info(f'total co-citation count: {len(cocits_set):,}')

    return cocits_set


def get_authors(doc_id, doc_index):
    if doc_id in doc_index:
        s2paper = doc_index[doc_id]
        last_names = [a['name'].split()[-1].lower() for a in s2paper['authors']]
        return last_names
    else:
        raise ValueError(f'Doc ID not found: {doc_id}')


def have_no_shared_authors(a_id, b_id, doc_index):
    try:
        a_authors = set(get_authors(a_id, doc_index))
        b_authors = set(get_authors(b_id, doc_index))

        overlap = a_authors & b_authors

        if len(overlap) == 0:
            return True
        else:
            return False

    except ValueError:
        return False


# has same venue
def get_venue(doc_id, doc_index):
    if doc_id in doc_index:
        s2paper = doc_index[doc_id]
        return s2paper['venue'].lower().strip()
    else:
        raise ValueError(f'Doc ID not found: {doc_id}')


def have_not_same_venue(a_id, b_id, doc_index):
    a_venue = get_venue(a_id, doc_index)
    b_venue = get_venue(b_id, doc_index)

    if a_venue == "" or b_venue == "":
        # cant answer if venue is not set
        return False

    if fuzz.ratio(a_venue, b_venue) < 0.75:
        # fuzzy string matching score must be low!
        return True
    else:
        return False


def get_negative_pairs(s2_id2s2_paper, positive_pairs, cits_set, cocits_set, negative_ratio=0.5, negative_count=0):
    # negative_label = 'none'
    # negative_needed = 10000 #105492  # len(df)

    if negative_count > 0:
        negative_needed = negative_count
    else:
        negative_needed = math.ceil(len(positive_pairs) * negative_ratio)

    # negative_rows = []
    negative_pairs = set()
    tries = 0
    all_doc_ids = list(s2_id2s2_paper.keys())

    logger.info(f'Negatives needed: {negative_needed:,} (ratio: {negative_ratio}, fixed: {negative_count})')

    while len(negative_pairs) < negative_needed:
        a = random.choice(all_doc_ids)
        b = random.choice(all_doc_ids)

        if a == b:
            tries += 1
            continue

        if not have_no_shared_authors(a, b, s2_id2s2_paper):
            tries += 1
            continue

        if not have_not_same_venue(a, b, s2_id2s2_paper):
            tries += 1
            continue

        cit_pair = get_sorted_pair(a, b)
        if cit_pair in cits_set:
            tries += 1
            continue

        if cit_pair in cocits_set:
            tries += 1
            continue

        # text = get_text_from_doc_id(a, s2_id2s2_paper)
        # text_b = get_text_from_doc_id(b, s2_id2s2_paper)
        #
        # if text == '' or text_b == '':
        #     continue

        pair = tuple((a, b))

        if pair in negative_pairs:
            continue

        negative_pairs.add(pair)

        # negative_rows.append((
        #     text,
        #     text_b,
        #     negative_label,
        # ))

    logger.info(f'Found {len(negative_pairs):,} negative rows (tried {tries:,} random samples)')

    return negative_pairs