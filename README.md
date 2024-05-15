# Quantized Instruction-Tuned Large Language Models for Zero-Shot SDG Classification

## Overview
This repository contains the code accompanying the paper "Quantized Instruction-Tuned Large Language Models for Zero-Shot SDG Classification". This paper investigates the potential of quantized, instruction-tuned Large Language Models (LLMs) for zero-shot classification of scientific abstracts according to the United Nations' Sustainable Development Goals (SDGs). We introduce the Decompose-Synthesize-Refine-Extract (DSRE) framework, leveraging advanced prompting techniques for both single-label and multi-label classification scenarios.

## Structure
- `src/`: Contains the source code for the DSRE framework and the various experiments.
- `data/`: (optional) Directory for dataset storage, not included in the repository due to size.
- `notebooks/`: Jupyter notebooks used for experiments and analysis.
- `results/`: Directory to save experimental results, including performance metrics and logs.
- `trained_adapters/`: Directory to save trained adapter modules.
- `pyproject.toml`: Poetry configuration file.
- `poetry.lock`: Lock file for the dependencies.

## Installation
```bash
git clone https://github.com/TobiFank/SDG-Classification-Using-Instruction-Tuned-LLMs.git
cd SDG-Classification-Using-Instruction-Tuned-LLMs
```

This project uses [Poetry](https://python-poetry.org/) for dependency management. Ensure you have Poetry installed, then run the following commands to set up the environment:

```bash
poetry install
```

You need to have access to a huggingface api key which has been approved by meta to use the llama weights. Add your key in the config.py file in the src directory.
Also having a GPU with CUDA 12.2 is the tested configuration.

## Usage

```bash
poetry shell
```

- To create the prompts run the notebook 'prompt_generator.ipynb' in the 'notebooks' directory. The prompts are saved in the 'data/prompts' directory. 
- To run the DSRE run the 'DSRE.ipynb' notebook in the 'notebooks' directory. The results are saved in the 'results' directory.
- To train the LLAMA or Zephyr model run the corresponding python file in the 'src' directory.
```bash
python src/train_llama.py
```
Hyperparameters can be adjusted using e.g.
```bash
python src/train_llama.py --batch_size 8 --train_on zo_up
```

- For inference LLAMA or Zepry models run the corresponding python file in the 'src' directory.
```bash
python src/inference_llama.py
```
Hyperparameters can be adjusted using e.g. the adapter trained with zo_up data.
```bash
python src/inference_llama.py --trained_on zo_up
```

- To train the extraction adapter for the DSRE run the 'train_DSRE_extraction.py' file in the 'src' directory e.g. for multi-label classification:
```bash
python src/train_DSRE_extraction.py --task multi
```
