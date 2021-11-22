# Aspect-based Document Similarity for Research Papers

Implementation, trained models and result data for the paper **Aspect-based Document Similarity for Research Papers** [(PDF on Arxiv)](https://arxiv.org/abs/2010.06395). 
The supplemental material is available for download under [GitHub Releases](https://github.com/malteos/aspect-document-similarity/releases) or [Zenodo](http://doi.org/10.5281/zenodo.4087898).

- Datasets are compatible with 🤗 [Huggingface NLP library](https://github.com/huggingface/nlp) (now known as [datasets](https://github.com/huggingface/datasets)). 
- Models are available on 🤗 [Huggingface Transformers models](https://huggingface.co/malteos). 

<img src="https://raw.githubusercontent.com/malteos/aspect-document-similarity/master/docrel.png">

## Demo

<a href="https://colab.research.google.com/github/malteos/aspect-document-similarity/blob/master/demo.ipynb"><img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Google Colab"></a>

You can try our trained models directly on Google Colab on all papers available on Semantic Scholar (via DOI, ArXiv ID, ACL ID, PubMed ID):

<a href="https://colab.research.google.com/github/malteos/aspect-document-similarity/blob/master/demo.ipynb"><img src="https://raw.githubusercontent.com/malteos/aspect-document-similarity/master/demo.gif" alt="Click here for demo"></a>

## Requirements

- Python 3.7
- CUDA GPU (for Transformers)

Datasets
- [ACL Anthology Reference Corpus (ACL ARC)](http://acl-arc.comp.nus.edu.sg/)
- [COVID-19 Open Research Dataset (CORD 19)](https://www.semanticscholar.org/cord19)

## Installation

Create a new virtual environment for Python 3.7 with Conda:
 
 ```bash
conda create -n paper python=3.7
conda activate paper
```

Clone repository and install dependencies:
```bash
git clone https://github.com/malteos/aspect-document-similarity.git repo
cd repo
pip install -r requirements.txt
```

## Experiments

To reproduce our experiments, follow these steps (if you just want to train and test the models, skip the first two steps):

### Prepare

```bash
export DIR=./output

# ACL Anthology 
# Get parscit files from: https://acl-arc.comp.nus.edu.sg/archives/acl-arc-160301-parscit/)
sh ./sbin/download_parsecit.sh

# CORD-19
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-03-13.tar.gz

# Get additional data (collected from Semantic Scholar API)
wget https://github.com/malteos/aspect-document-similarity/releases/download/1.0/acl_s2.tar
wget https://github.com/malteos/aspect-document-similarity/releases/download/1.0/cord19_s2.tar
```

### Build datasets

```bash
# ACL
python -m acl.dataset save_dataset <input_dir> <parscit_dir> <output_dir>

# CORD-19
python -m cord19.dataset save_dataset <input_dir> <output_dir>

```

### Use dataset

The datasets are built on the Huggingface NLP library (soon available on the official repository):

```python
from nlp import load_dataset

# Training data for first CV split
train_dataset = load_dataset(
    './datasets/cord19_docrel/cord19_docrel.py',
    name='relations',
    split='fold_1_train'
)                   
```

### Train models

All models are trained with the `trainer_cli.py` script:

```bash
python trainer_cli.py --cv_fold $CV_FOLD \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME \
    --doc_id_col $DOC_ID_COL \
    --doc_a_col $DOC_A_COL \
    --doc_b_col $DOC_B_COL \
    --nlp_dataset $NLP_DATASET \
    --nlp_cache_dir $NLP_CACHE_DIR \
    --cache_dir $CACHE_DIR \
    --num_train_epochs $EPOCHS \
    --seed $SEED \
    --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LR \
    --do_train \
    --save_predictions
```

The exact parameters are available in `sbin/acl` and `sbin/cord19`. 



### Evaluation

The results can be computed and viewed with a Jupyter notebook. 
Figures and tables from the paper are part of the notebook.

```bash
jupyter notebook evaluation.ipynb
```

Due to the space constraints some results could not be included in the paper.
The full results for all methods and all test samples are available as 
CSV files under `Releases`
(or via the Jupyter notebook).

## How to cite

If you are using our code, please cite [our paper](https://arxiv.org/abs/2010.06395):

```bibtex
@InProceedings{Ostendorff2020c,
  title = {Aspect-based Document Similarity for Research Papers},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)},
  author = {Ostendorff, Malte and Ruas, Terry and Blume, Till and Gipp, Bela and Rehm, Georg},
  year = {2020},
  month = {Dec.},
}
```

## License

MIT


