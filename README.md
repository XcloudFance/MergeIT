# MergeIT
MergeIT: From Selection to Merging for Efficient Instruction Tuning


# Overview
Instruction tuning is crucial for optimizing Large Language Models (LLMs), yet mainstream data selection methods heavily rely on LLMs as instruction quality scorers, leading to high computational costs and reduced data diversity. To address these limitations, we propose **MergeIT**, a novel LLM-based **Merging** strategy for better **I**nstruction **T**uning that shifts the focus from selection to synthesis. MergeIT operates in two stages: first, topic-aware filtering clusters and refines the dataset, preserving diversity while eliminating redundancy without relying on LLM-based scoring. Second, LLM-based merging synthesizes semantically similar instructions into more informative and compact training data, enhancing data richness while further reducing dataset size. Experimental results demonstrate that MergeIT enables efficient, diverse, and scalable instruction selection and synthesis, establishing LLM-based merging as a promising alternative to conventional scoring-based selection methods for instruction tuning.

# Install
For installation, run the following commands in the bash.

```bash
git clone https://github.com/XcloudFance/MergeIT
pip3 install -r requirements.txt
git clone https://github.com/hkust-nlp/deita.git
cd deita
pip install -e .
cd ..
```

# Quick Start
1. Change the `diverse.py` filepath into your own IFT dataset files.
```bash
python diverse.py
```
2. After running `diverse.py`, an inital dataset is well-prepared for MergeIT to merge the final subset.
```bash
python merging.py
```
Make sure all filenames are aligned before you start running them.

3. Evaluation
Kindly refer to https://github.com/EleutherAI/lm-evaluation-harness/, which we use for the benchmarks in our entire paper.


# Dataset
We release `MergeIT-6k.json` as our subset data for instruction tuning. 

# Acknowledgement
https://github.com/hkust-nlp/deita
