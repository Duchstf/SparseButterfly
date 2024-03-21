# SparseButterfly
Butterfly matrix for Sparse NN Inference on FPGA

Follow the instructions below to run our codes:

## 1. Conda environment

Create conda environment:

```
conda-env create -f environment.yml
```

Activate the environment:

```
conda activate fly
```

install pytorch:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

And then do whatever you want in this environment (edit files, open notebooks, etc.). To deactivate the environment:

```
conda deactivate
```

If you make any update for the environment, please edit the `environment.yml` file and run:

```
conda env update --file environment.yml  --prune
```

Reference on conda environment here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


# Related Code

* Pixelfly: https://github.com/HazyResearch/fly
* Sparse Linear Networks with a Fixed Butterfly Structure: Theory and Practice. https://github.com/leibovit/Sparse-Linear-Networks

# Related Blog Posts

Butterflies Are All You Need: A Universal Building Block for Structured Linear Maps: https://dawn.cs.stanford.edu/2019/06/13/butterfly/

# Related Paper

* Monarch: Expressive Structured Matrices for Efficient and
Accurate Training: https://arxiv.org/pdf/2204.00595.pdf

* Sparse Linear Networks with a Fixed Butterfly Structure: Theory and Practice: https://proceedings.mlr.press/v161/ailon21a/ailon21a.pdf

* Scatterbrain: Unifying Sparse and Low-rank Attention
Approximation: https://arxiv.org/pdf/2110.15343.pdf

* Pixelated Butterfly: Simple and Efficient Sparse Training for
Neural Network Models: https://arxiv.org/pdf/2112.00029.pdf

* Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations: https://arxiv.org/abs/1903.05895

* Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design: https://steliosven10.github.io/papers/[2022]_micro_adaptable_butterfly_accelerator_for_attention_based_nns_via_hardware_and_algorithm_codesign.pdf
* Deformable Butterfly: A Highly Structured and Sparse Linear Transform: https://arxiv.org/pdf/2203.13556.pdf

