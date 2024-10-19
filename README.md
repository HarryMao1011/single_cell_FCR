# Learning Identifiable Factorized Causal Representations of Cellular Responses

[OpenReview](https://openreview.net/forum?id=AhlaBDHMQh) |
[arXiv](https://arxiv.org/abs/***) |
[BibTeX](#bibtex)

<p align="center">
    <img alt="Learning Identifiable Factorized Causal Representations of Cellular Responses" src="assets/examples.png" width="500">
</p>

Official code for the NeurIPS 2024 paper [ Learning Identifiable Factorized Causal Representations of Cellular Responses](https://openreview.net/forum?id=AhlaBDHMQh). This work was performed by
[Haiyi Mao](https://harrymao1011.github.io/),
[Romain Lopez](https://romain-lopez.github.io/),
[Kai liu](),
[Panayiotis (Takis) Benos](),
[Qiu Lin](https://lquvatexas.github.io/),
Please [cite us](#bibtex) when making use of our code or ideas.

## Installation
<p align="left">
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://pytorch.org/get-started/previous-versions/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.3.1-orange.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://anaconda.org/anaconda/conda"><img alt="virtualenv" src="https://img.shields.io/badge/virtualenv-conda-green.svg"></a>
</p>

```shell
cd $PROJECT_DIR
conda config --append channels conda-forge
conda create -n fcr-env --file requirements.txt
conda activate fcr-env
```


## Data Availability
sciPlex: 


## Run
```shell
# train
python main.py

# evaluate
python main_numerical --evaluate
```

## License
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
<!-- `[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)` -->
#### Attribution-NonCommercial-NoDerivatives 4.0 International
[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)  
<!-- `[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)`   -->


<!-- ## Multimodal Experiment

Download the dataset [Multimodal3DIdent](https://zenodo.org/records/7678231) [Daunhawer et al. ICLR 2023]:
```shell
# download and extract the dataset
$ wget https://zenodo.org/record/7678231/files/m3di.tar.gz
$ tar -xzf m3di.tar.gz
```
Training and evaluation:
```shell
# train a model with three input views (img0, img1, txt0)
python main_multimodal.py --dataroot "$PATH_TO_DATA"  --dataset "multimodal3di"

# evaluate
python main_multimodal --dataroot "$PATH_TO_DATA" --dataset "multimodal3di" --evaluate
```
# Acknowledgements
This implementation is built upon [multimodal-contrastive-learning](https://github.com/imantdaunhawer/multimodal-contrastive-learning) and [ssl_identifiability](https://github.com/ysharma1126/ssl_identifiability). -->

## BibTex

<!-- ```bibtex
@inproceedings{
    yao2024multiview,
    title={Multi-View Causal Representation Learning with Partial Observability},
    author={Dingling Yao and Danru Xu and S{\'e}bastien Lachapelle and Sara Magliacane and Perouz Taslakian and Georg Martius and Julius von K{\"u}gelgen and Francesco Locatello},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=OGtnhKQJms}
}
``` -->
