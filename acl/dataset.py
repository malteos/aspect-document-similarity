import json
import logging
import os
import sys
from pathlib import Path

import fire
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from smart_open import open
from tqdm import tqdm

from acl.preprocessing.negative_sampling import get_cocitations, get_negative_pairs
from acl.preprocessing.parsecit import get_parsecit_files, get_citation_pairs_from_parscit
from acl.utils import resolve_and_sect_titles, to_label, get_sorted_pair, get_text_from_doc, \
    normalize_title

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def save_dataset(input_dir, parscit_dir, output_dir, cv_folds=4):
    """

    Run with: $ python -m acl.dataset save_dataset <input_dir> <parscit_dir> <output_dir>

    Required parscit directory (from ACL-anthology):
    - Download and extract from: https://acl-arc.comp.nus.edu.sg/archives/acl-arc-160301-parscit/
    - parscit/A/A00/A00-1000-parscit.130908.xml
    - ...

    Required input files (.json or .json.gz):
    - title2dblp_hits.json
    - acl_id2s2.json
    - arxiv2s2.json
    - doi2s2.json.gz

    Output structure
    - docs.jsonl: each line is a S2-paper
    - folds/1/
    - folds/2/
    - ...
    - folds/k/train.csv: actual training samples
    - folds/k/test.csv

    Samples are provided as CSV files with the following columns:
    - doc_a: S2-id
    - doc_b: S2-id
    - label: List of labels (comma separated)

    After dataset creation use the following commands to compress and upload all files:

    cd <output_dir>
    tar -cvzf acl_docrel.tar.gz docs.jsonl folds/
    curl --upload-file acl_docrel.tar.gz ftp://$FTP_LOGIN:$FTP_PASSWORD@$FTP_HOST/$FTP_DIR

    :param input_dir: S2 paper files
    :param output_dir: Dataset files written to this directory
    :param parscit_dir:
    :param cv_folds:
    :return:
    """
    negative_label = 'none'
    min_text_length = 50
    negative_sampling_ratio = 0.5

    # Fixed labels
    """
    introduction     20515
    related work     14883
    experiment        5749
    conclusion        1914
    results           1828
    background        1748
    discussion        1627
    evaluation        1386
    method             927
    (previous work      902)
    """
    labels = [
        'introduction',
        'related work',
        'experiment',
        'conclusion',
        'results',
        'background',
        'discussion',
        'evaluation',
        'method',
        # Only top-9 label classes for v1.1 (equal to CORD-19)
        # 'previous work'
    ]

    doc_a_col = 'from_s2_id'
    doc_b_col = 'to_s2_id'
    label_col = 'label'

    # Convert dirs to Path if is string
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    # Load paper data from various sources
    # acl_id2title, doi2title, arxiv2title = get_dblp_titles(input_dir / 'title2dblp_hits.json.gz')  # TODO
    acl_id2s2 = json.load(open(input_dir / 'acl_id2s2.json.gz', 'r'))
    arxiv2s2 = json.load(open(input_dir / 'arxiv2s2.json.gz', 'r'))
    doi2s2 = json.load(open(input_dir / 'doi2s2.json.gz', 'r'))

    # Merge S2 data
    s2_id2s2_paper = {}
    s2_id2s2_paper.update({s2['paperId']: s2 for _id, s2 in acl_id2s2.items()})
    s2_id2s2_paper.update({s2['paperId']: s2 for _id, s2 in arxiv2s2.items()})
    s2_id2s2_paper.update({s2['paperId']: s2 for _id, s2 in doi2s2.items()})

    # Filter by empty text
    s2_id2s2_paper = {s2_id: p for s2_id, p in s2_id2s2_paper.items() if len(get_text_from_doc(p)) >= min_text_length}

    # Title mapping from document index
    title2s2_id = {normalize_title(p['title']): s2_id for s2_id, p in s2_id2s2_paper.items()}

    parscit_files = get_parsecit_files(parscit_dir)
    cit_pairs, error_files = get_citation_pairs_from_parscit(parscit_files, acl_id2s2, title2s2_id)

    # s2_pairs, s2_pairs_not_found = get_s2_pairs_from_cits(cit_pairs, acl_id2s2)
    normalized_s2_pairs = resolve_and_sect_titles(cit_pairs, doc_index=s2_id2s2_paper)

    # Convert to dataframe
    df = pd.DataFrame(normalized_s2_pairs, columns=['from_s2_id', 'to_s2_id', 'citing_section', 'marker'])

    # Auto-determine top labels
    pre_label_col = 'citing_section'
    # top_sections = 10
    # labels = list(filter(lambda t: t, df[pre_label_col].value_counts()[:top_sections].keys()))

    # Remove duplicates
    logger.info(f'Before drop duplications: {len(df)}')

    df[label_col] = [to_label(t, labels) for t in df[pre_label_col]]
    df.drop_duplicates([doc_a_col, doc_b_col, label_col], keep='first', inplace=True)

    logger.info(f'After drop duplications: {len(df)}')

    # join multi-labels
    # df = df.groupby([doc_a_col, doc_b_col]).label.agg(
    #     [('label_count', 'count'), (label_col, ','.join)]).reset_index()
    df = df.groupby([doc_a_col, doc_b_col]).label.agg(
        [(label_col, ','.join)]).reset_index()

    # Positive samples
    # pos_rows = []
    #
    # for idx, r in df.iterrows():
    #     text = get_text_from_doc_id(r[doc_a_col], s2_id2s2_paper)
    #     text_b = get_text_from_doc_id(r[doc_b_col], s2_id2s2_paper)
    #
    #     # Filter out empty texts
    #     if text != '' and text_b != '':
    #         pos_rows.append((text, text_b, r[label_col]))
    cits_list = df[[doc_a_col, doc_b_col]].values.tolist()
    cits_set = {get_sorted_pair(from_id, to_id) for from_id, to_id in cits_list}

    logger.info(f'Total citation count: {len(cits_set):,}')

    # co cits
    cocits_set = get_cocitations(df[[doc_a_col, doc_b_col]].values.tolist())

    # Negative sampling
    negative_pairs = get_negative_pairs(s2_id2s2_paper, cits_list, cits_set, cocits_set,
                                        negative_ratio=negative_sampling_ratio)

    # construct dataset frame
    logger.info('Constructing dataset data frame...')
    dataset = df[[doc_a_col, doc_b_col, label_col]].values.tolist()\
        + list(map(lambda p: (p[0], p[1], negative_label), negative_pairs))  # positive + negative pairs

    dataset_df = pd.DataFrame(dataset, columns=[doc_a_col, doc_b_col, label_col])

    # Verify
    missing_doc_ids = [doc_id for doc_id in dataset_df[doc_a_col].values if doc_id not in s2_id2s2_paper]
    missing_doc_ids += [doc_id for doc_id in dataset_df[doc_b_col].values if doc_id not in s2_id2s2_paper]

    if len(missing_doc_ids) > 0:
        raise ValueError(f'Document IDs are missing in index: {missing_doc_ids}')

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
        for paper in tqdm(s2_id2s2_paper.values(), desc='Writing document data', total=len(s2_id2s2_paper)):
            f.write(json.dumps(paper) + '\n')

    logger.info('Done')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
