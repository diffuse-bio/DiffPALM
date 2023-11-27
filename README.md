# DiffPALM

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

Modified Instructions for internal use at Diffuse Bio

Inspite of the large number of MSA samples provided as examples with the original
code (about 25K), only about 50 to 100 pairs of MSA and less than 10 species are
used by the system. A T4 GPU can support apprximately around 100 MSA pairs.

To do: MSA sequences / species need to be prioritized.


Below are the instructions for the original repo.

Paralog matching method described in [“Pairing interacting protein
sequences using masked language modeling” (Lupo, Sgarbossa, and Bitbol,
2023)](https://www.biorxiv.org/content/10.1101/2023.08.14.553209). The
MSA Transformer model used here was introduced in [(Rao el al,
2021)](https://proceedings.mlr.press/v139/rao21a.html).

## Install

Clone this repository on your local machine by running and move inside
the root folder. We recommend creating and activating a dedicated conda
or virtualenv Python virtual environment.

``` sh
git clone git@github.com:Bitbol-Lab/DiffPALM.git
```

and move inside the root folder. We recommend creating and activating a
dedicated conda or virtualenv Python virtual environment. Then, make an
editable install of the package:

``` sh
python -m pip install -e .
```

## How to use

See the
[`_example_prokaryotic.ipynb`](https://github.com/Bitbol-Lab/DiffPALM/blob/main/nbs/_example_prokaryotic.ipynb)
notebook for an example of paired MSA optimization in the case of
well-known prokaryotic datasets, for which ground truth matchings are
given by genome proximity.

## Citation

Our work can be cited using the following BibTeX entry:

``` bibtex
@article{lupo2023pairing,
  title={Pairing interacting protein sequences using masked language modeling},
  author={Lupo, Umberto and Sgarbossa, Damiano and Bitbol, Anne-Florence},
  year={2023},
  journal={bioRxiv},
  doi={10.1101/2023.08.14.553209 }
}
```

## nbdev

This project has been developed using [nbdev](https://nbdev.fast.ai/).
