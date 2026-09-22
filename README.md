# GPT-2 Multitask Learning

Course-project experiments adapting a compact GPT-2 implementation to several downstream NLP tasks, with an emphasis on learning under limited data and compute.

## Project overview

The project covers:

- sentiment classification on SST and CFIMDB;
- paraphrase detection;
- sonnet generation with configurable decoding;
- comparisons between linear probing and full-model fine-tuning;
- regularization and hyperparameter-search experiments;

Most source files originate from a course starter repository. The course-project work focused on completing and extending the GPT-2 implementation, optimizer and training workflow, then designing and running the downstream experiments. See [ATTRIBUTION.md](ATTRIBUTION.md) for provenance and licensing details.

## Repository layout

```text
.
├── models/                     # GPT-2 model implementation
├── modules/                    # attention and transformer-layer modules
├── results/                    # sanitized experiment summary
├── classifier.py               # sentiment-classification entry point
├── classifier_hpsearch.py      # hyperparameter search
├── classifier_hps2.py          # extended experiment configuration
├── classifier_hps3.py          # extended experiment configuration
├── paraphrase_detection.py     # paraphrase task
├── sonnet_generation.py        # language-generation task
├── optimizer.py                # AdamW implementation
└── env.yml                     # Conda environment
```

Datasets, checkpoints, generated predictions and other large artifacts are intentionally excluded.

## Setup

Create the environment:

```bash
conda env create -f env.yml
conda activate cs224n_dfp
```

Obtain the datasets through the original course instructions and place them in `data/`. The scripts expose their available options through `--help`, for example:

```bash
python classifier.py --help
python paraphrase_detection.py --help
python sonnet_generation.py --help
```

## Selected observations

- Full-model fine-tuning consistently outperformed linear probing in the sentiment experiments.
- The smaller SST dataset was more sensitive to overfitting than CFIMDB.
- Controlled sampling and repetition handling improved the quality of generated sonnets.

Detailed run-level metrics are available in [`results/experiment-summary.csv`](results/experiment-summary.csv). This file omits local paths, timestamps and checkpoint names.

## License

The inherited starter code remains under the Apache License 2.0. See [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md).
