# PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models

Tadeusz Dziarmaga, Marcin Kądziołka, Artur Kasymov, Marcin Mazur

This code accompanies the paper: PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models

The code has been tested with Python 3.13.2 on Linux.

Abstract: *Deep generative models (DGMs) have caused a paradigm shift in the field 
of machine learning, yielding noteworthy advancements in domains such as
 image synthesis, natural language processing, and other related areas. 
However, a comprehensive evaluation of these models that accounts for 
the trichotomy between fidelity, diversity, and novelty in generated 
samples remains a formidable challenge. A recently introduced solution 
that has emerged as a promising approach in this regard is the Feature 
Likelihood Divergence (FLD), a method that offers a theoretically 
motivated practical tool, yet also exhibits some computational 
challenges. In this paper, we propose PALATE, a novel enhancement to the
 evaluation of DGMs that addresses limitations of existing metrics. Our 
approach is based on a peculiar application of the law of total 
expectation to random variables representing accessible real data. When 
combined with the MMD baseline metric and DINOv2 feature extractor, 
PALATE offers a holistic evaluation framework that matches or surpasses 
state-of-the-art solutions while providing superior computational 
efficiency and scalability to large-scale datasets. Through a series of 
experiments, we demonstrate the effectiveness of the PALATE enhancement,
 contributing a computationally efficient, holistic evaluation approach 
that advances the field of DGMs assessment, especially in detecting 
sample memorization and evaluating generalization capabilities.*

## Instructions to Run

### 1. Setting things up

#### a. Install conda

Install one of Anaconda Distributions (for example [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)).

#### b. Create conda environment:

```bash
conda env create -f environment.yml
```

#### c. Activate environment:

```bash
conda activate palate
```

### 2. Example call

```bash
python3 main.py ./path/to/train ./path/to/test ./path/to/gen_1 ./path/to/gen_2 --batch_size 256 --nsample 1000 --save --load
```

- The **first path** should point to the **training data**.

- The **second path** should point to the **test data**.

- Subsequent paths should point to folders containing **generated samples** (one folder per model).

Each run generates a unique folder in the specified output directory. The folder contains metrics summary in `.txt` and `.csv` format.

### Detailed Information

```text
usage: main.py [-h] [--model {dinov2}] [-bs BATCH_SIZE] [--num-workers NUM_WORKERS] [--device DEVICE] [--nsample NSAMPLE] [--output_dir OUTPUT_DIR] [--save] [--load] [--no-load] [--seed SEED] [--clean_resize] [--depth DEPTH]  
               [--repr_dir REPR_DIR]
               path [path ...]

positional arguments:
  path                  Paths to the images. The order is train, test, gen_1, gen_2, ..., gen_n

options:
  -h, --help            show this help message and exit
  --model {dinov2}      Model to use for generating feature representations. (default: dinov2)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size to use (default: 50)
  --num-workers NUM_WORKERS
                        Number of processes to use for data loading. Defaults to `min(8, num_cpus)` (default: None)
  --device DEVICE       Device to use. Like cuda, cuda:0 or cpu (default: None)
  --nsample NSAMPLE     Maximum number of images to use for calculation (default: 10000)
  --output_dir OUTPUT_DIR
                        Directory for saving outputs: metrics_summary.csv, metrics_summary.txt and arguments.txt (default: output/)
  --save                Save representations to repr_dir (default: False)
  --load                Load representations and statistics from previous runs if possible (default: True)
  --no-load             Do not load representations and statistics from previous runs. (default: True)
  --seed SEED           Random seed (default: 13579)
  --clean_resize        Use clean resizing (from pillow) (default: False)
  --depth DEPTH         Negative depth for internal layers, positive 1 for after projection head. (default: 0)
  --repr_dir REPR_DIR   Dir to store saved representations. (default: ./saved_representations)
```

## Datasets

All of the data samples we use are provided by [Stein et al [2023]](https://arxiv.org/abs/2306.04675), who have open-sourced a set of samples from a variety of SOTA models on different datasets.

### CIFAR-10 Experiments

For all CIFAR-10 experiments, we use training, testing and pre-generated samples these can be accessed from [this Google Drive folder](https://drive.google.com/drive/folders/1r38UxZQcREoNklnzQFj9kxUXMqVsceOQ).

#### Data Loading

The dataloader automatically handles both conditional and unconditional models:

- Conditional models: Data is organized into folders labeled `0` to `9`, representing the 10 classes.

- Unconditional models: Data is stored in a single folder.

To run the script provide the path to the folder containing the model's data.

### ImageNet Experiments

For all ImageNet experiments, we use training and testing data that can be accessed from [imagenet256.zip](https://drive.google.com/file/d/1kbAvWrXw_GCE4ulMlx5zlfVOG_v6iMM7/view?usp=drive_link). We use pre-generated samples from [this Google Drive folder](https://drive.google.com/drive/folders/11uSNfzKfH0GfTtS1fgowQ_HNdof2gSZT).

#### Data Loading

Data loading is handled in the same way as CIFAR-10.

## BibTeX

```
@Article{dziarmaga2025palate,
        author={Tadeusz Dziarmaga and Marcin Kądziołka and Artur Kasymov and Marcin Mazur},
        year={2025},
        title={PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models},
        eprint={2503.18462},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
}
```

