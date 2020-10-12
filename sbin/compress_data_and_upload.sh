#!/usr/bin/env bash

# Compress data for upload

# ACL: S2 papers
tar -cvf acl_s2.tar title2dblp_hits.json.gz acl_id2s2.json.gz arxiv2s2.json.gz doi2s2.json.gz

# CORD-19
tar -cvf cord19_s2.tar metadata.csv doi2s2paper.json.gz
bzip2 cord19_s2.tar

# Models (SciBERT)
tar -cvzf ./cord19_fold-1_scibert-scivocab-uncased.tar.gz  --directory=../output/cord19_docrel/folds/1/ scibert-scivocab-uncased
tar -cvzf ./acl_fold-1_scibert-scivocab-uncased.tar.gz  --directory=../output/acl_docrel/folds/1/ scibert-scivocab-uncased

# Results
tar -cvzf acl_output.tar.gz --exclude='*.bin' --exclude='__*' ../output/acl_docrel/*
tar -cvzf cord19_output.tar.gz --exclude='*.bin' --exclude='__*' ../output/cord19_docrel/*


### Upload to GitHub release (with https://github.com/github-release/github-release)
export GITHUB_TOKEN=
export GITHUB_USER=
export GITHUB_REPO=

~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name acl_s2.tar --file acl_s2.tar
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name cord19_s2.tar.bz2 --file cord19_s2.tar.bz2
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name acl_fold-1_scibert-scivocab-uncased.tar.gz --file acl_fold-1_scibert-scivocab-uncased.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name cord19_fold-1_scibert-scivocab-uncased.tar.gz --file cord19_fold-1_scibert-scivocab-uncased.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name acl_output.tar.gz  --file acl_output.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name cord19_output.tar.gz  --file cord19_output.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name scibert-vocab.txt --file ~/datasets/BERT_pre_trained_models/pytorch/scibert-scivocab-uncased/vocab.txt

