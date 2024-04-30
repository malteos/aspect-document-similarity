import json
import logging
import os
import sys
from pathlib import Path
from typing import Union

import fire
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from smart_open import open
from tqdm import tqdm

from acl.preprocessing.negative_sampling import get_cocitations
from acl.utils import get_sorted_pair, to_label
from cord19.preprocessing.cord19_reader import get_papers_and_citations_from_cord19, merge_cord19_and_s2_papers
from cord19.preprocessing.negative_sampling import get_negative_pairs
from cord19.utils import normalize_section, resolve_and_sect_titles, get_text_from_doi

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def save_dataset(input_dir: Union[str, Path], output_dir: Union[str, Path], cv_folds: int = 4):
    """

    Run with: $ python -m cord19.dataset save_dataset <input_dir> <output_dir>

    input_dir = '/home/mostendorff/datasets/cord-19/'
    output_dir = '/home/mostendorff/datasets/cord-19/dataset/'
    cv_folds = 4

    input_dir/metadata.csv
    input_dir/doi2paper.json.gz
    input_dir/<subsets> = ['biorxiv_medrxiv', 'comm_use_subset', 'custom_license', 'noncomm_use_subset']

    output_dir/docs.jsonl
    output_dir/folds/1/train.csv
    output_dir/folds/1/test.csv

    tar -cvzf cord19_docrel.tar.gz docs.jsonl folds/
    curl --upload-file cord19_docrel.tar.gz ftp://$FTP_LOGIN:$FTP_PASSWORD@ostendorff.org/cloud.ostendorff.org/static/

    :param input_dir: Path to directory with input files
    :param output_dir: Output files are written to this dir
    :param cv_folds: Number of folds in k-fold cross validation
    """
    label_col = 'label'
    negative_label = 'none'
    min_text_length = 50
    negative_sampling_ratio = 0.5

    doc_a_col = 'from_doi'
    doc_b_col = 'to_doi'

    labels = [
        'discussion',
        'introduction',
        'conclusion',
        'results',
        'methods',
        'background',
        'materials',
        'virus',
        'future work'
    ]

    # input_dir = os.path.join(env['datasets_dir'], 'cord-19')

    # Convert dirs to Path if is string
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    # Read meta data
    meta_df = pd.read_csv(input_dir / 'metadata.csv', index_col=0, dtype={'doi': str, 'journal': str})
    id2meta = {row['sha']: row for idx, row in meta_df.iterrows() if row['sha']}

    logger.info('Unique DOIs in meta data: %s' % (len(meta_df['doi'].unique()) / len(meta_df)))

    # Load paper data and citations from CORD-19
    id2paper, cits = get_papers_and_citations_from_cord19(input_dir, id2meta)

    # Load paper data from disk (scraped from S2)
    if os.path.exists(input_dir / 'doi2s2paper.json.gz'):
        with open(str(input_dir / 'doi2s2paper.json.gz'), 'r') as f:
            doi2s2paper = json.load(f)

        logger.info(f'Loaded {len(doi2s2paper):,} scraped papers from disk')
    else:
        logger.error('Cannot load S2 papers from: %s' % (input_dir / 'doi2paper.json.gz'))
        doi2s2paper = {}

    # Merge CORD-19 papers and S2 papers
    doi2paper = merge_cord19_and_s2_papers(id2paper, id2meta, doi2s2paper)

    logger.info(f'Loaded {len(doi2paper)} from CORD-19')

    all_dois = list(doi2paper.keys())

    # DOIs with text
    doi2text = {}
    for doi in all_dois:
        text = get_text_from_doi(doi, doi2paper, raise_not_found_error=False)
        if len(text) > min_text_length:
            doi2text[doi] = text

    logger.info(f'Total DOIs: {len(all_dois):,}')
    logger.info(f'With text DOIs: {len(doi2text):,}')

    # Filter citations with existing DOI
    cits_with_doi = [c for c in cits if c[0] in doi2paper and c[1] in doi2paper]

    # CORD-19 only: Citations with DOI: 30655 (0.09342419246206499)
    # + S2: Citations with DOI: 170454 (0.5194756908148369)

    logger.info(f'Citations with DOI: {len(cits_with_doi)} ({len(cits_with_doi) / len(cits)})')

    missing_papers = [c[0] for c in cits if c[0] not in doi2paper]
    missing_papers += [c[1] for c in cits if c[1] not in doi2paper]

    logger.info(f'Missing paper data, but DOI: {len(missing_papers)}')

    unique_missing_papers = set(missing_papers)

    logger.info(f'Unique DOIs of missing papers: {len(unique_missing_papers)}')

    # resolve 'and' titles
    normalized_cits_with_doi = resolve_and_sect_titles(cits_with_doi)

    cits_df = pd.DataFrame(normalized_cits_with_doi, columns=[doc_a_col, doc_b_col, 'citing_section'])
    # cits_df

    logger.info(f'After normalization: {len(cits_df):,} (before: {len(cits_with_doi):,})')

    # top_sections = 10
    # labels = list(filter(lambda t: t, cits_df['citing_section'].value_counts()[:top_sections].keys()))

    # Remove duplicates
    cits_df[label_col] = [to_label(normalize_section(t), labels) for t in cits_df['citing_section']]
    cits_df.drop_duplicates([doc_a_col, doc_b_col, 'label'], keep='first', inplace=True)

    # Document must have text
    cits_df = cits_df[(cits_df[doc_a_col].isin(doi2text.keys())) & (cits_df[doc_a_col].isin(doi2text.keys()))]

    # Merge multi-labels
    df = cits_df.groupby([doc_a_col, doc_b_col]).label.agg([(label_col, ','.join)]).reset_index()

    # # Positive samples
    # pos_rows = []
    #
    # for idx, r in df.iterrows():
    #     text = get_text_from_doi(r[doc_a_col], doi2s2paper, doi2paper)
    #     text_b = get_text_from_doi(r[doc_b_col], doi2s2paper, doi2paper)
    #
    #     # Filter out empty texts
    #     if text != '' and text_b != '':
    #         pos_rows.append((r[doc_a_col], r[doc_b_col], text, text_b, r[label_col]))

    cits_set = set([get_sorted_pair(from_doi, to_doi) for from_doi, to_doi, label in cits_with_doi])

    logger.info(f'Total citation count: {len(cits_set):,}')

    cocits_set = get_cocitations([(from_doi, to_doi) for from_doi, to_doi, label in cits_with_doi])

    # Negatives needed: 52,746 (ratio: 0.5)
    negative_pairs = get_negative_pairs(
        doi2paper,
        candidate_doc_ids=list(doi2text.keys()),
        positive_pairs=df[[doc_a_col, doc_b_col]].values.tolist(),
        cits_set=cits_set,
        cocits_set=cocits_set,
        negative_ratio=negative_sampling_ratio
    )

    ###

    # construct dataset frame
    logger.info('Constructing dataset data frame...')
    dataset = df[[doc_a_col, doc_b_col, label_col]].values.tolist()\
        + list(map(lambda p: (p[0], p[1], negative_label), negative_pairs))  # positive + negative pairs

    dataset_df = pd.DataFrame(dataset, columns=[doc_a_col, doc_b_col, label_col])

    # TODO debug sample set?

    # Full training and test set
    logger.info(f'Creating {cv_folds}-Folds ')
    kf = StratifiedKFold(n_splits=cv_folds, random_state=0, shuffle=True)

    # Stratified K-Folds cross-validator
    for k, (train_index, test_index) in enumerate(
            kf.split(dataset_df.index.tolist(), dataset_df[label_col].values.tolist()), 1):
        fold_dir = os.path.join(output_dir, 'folds', str(k))

        if not os.path.exists(fold_dir):
            logger.info(f'Create new fold dir: {fold_dir}')
            os.makedirs(fold_dir)

        split_train_df = dataset_df.iloc[train_index]
        split_test_df = dataset_df.iloc[test_index]

        logger.info(f'Total: {len(dataset_df):,}; Train: {len(split_train_df):,}; Test: {len(split_test_df):,}')

        split_train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        split_test_df.to_csv(os.path.join(fold_dir, 'test.csv'),  index=False)

    # Write doc output
    with open(str(output_dir / 'docs.jsonl'), 'w') as f:
        for paper in tqdm(doi2paper.values(), desc='Writing document data', total=len(doi2paper)):
            f.write(json.dumps(paper) + '\n')

    logger.info('Done')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
