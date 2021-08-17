# ExpFinder: An Ensemble Model for Expert Finding from Text-based Expertise Data

## Introduction
<p align="justify">
Finding an expert plays a crucial role in driving successful collaborations and speeding up high-quality research development and innovations. However, the rapid growth of scientific publications and digital expertise data makes identifying the right experts a challenging problem. Existing approaches for finding experts given a topic can be categorised into information retrieval techniques based on vector space models, document language models, and graph-based models. In this paper, we propose <i>ExpFinder</i>, a new ensemble model for expert finding, that integrates a novel <i>N</i>-gram vector space model, denoted as <i>n</i>VSM, and a graph-based model, denoted as <i>μCO-HITS</i>, that is a proposed variation of the CO-HITS algorithm. The key of <i>n</i>VSM is to exploit recent inverse document frequency weighting method for <i>N</i>-gram words, and <i>ExpFinder</i> incorporates <i>n</i>VSM into <i>μCO-HITS</i> to achieve expert finding. 
</p>

## Setup steps
1. Clone the repository
```
git clone https://github.com/Yongbinkang/ExpFinder.git
```
2. Install dependencies
```
pip install requirements.txt
```
3. Copy and paste the command below for setting up Python path to execute file under the `scripts` directory
```
export PYTHONPATH="$PWD"
```
4. Download the SciBert model into the `model/` folder as mentioned in [this](https://github.com/Yongbinkang/ExpFinder/tree/main/model).

## Directory structure

![Directory structure](https://github.com/Yongbinkang/ExpFinder/blob/main/images/structure_interface.png)

For more instructions on setting up the project to run the pipeline in the `experimental pipline.ipynb` file, we sketch out the directory structure with description below:

* The __`data/`__ directory contains input or output data for the entire process. Four current data files in this directory is required for the data generation process.
* The __`model/`__ directory contains the SciBert model. Due to the large size of the model, we do not upload it here. For more details on how to download the model, please refer to the instruction at [this](https://github.com/Yongbinkang/ExpFinder/blob/main/model/README.md).
* The __`src/`__ directory contains the source code for the entire process including:
  * The __`algo/`__ directory has the `expfinder.py` file which is the source code for the ExpFinder algorithm. For more details about the algorithm, please refer to our paper.
  * The __`controller`__ directory has the `generator.py` file which is used to control the data generation process.
  * The __`lib`__ directory has four different python files serving different purposes as:
    * The `extractor.py` file aims to extract noun phrases from documents (using the `tokenization` module below).
    * The `vectorizer.py` file aims to vectorise every single phrase by using the SciBert model.
    * The `tokenization.py` file aims to extract tokens and noun phrases with their statistical information. Note that this contains the parser for the noun phrase extraction.
    * The `weight.py` file aims to calculate weights for given vectors or matrices (e.g., personalised weights or N-gram TFIDF).
* The __`scripts/`__ directory contains scripts for controlling all processes of a particular pipeline or demonstrating an example of a particular process.
* The __`experimental pipeline.ipynb`__ file contains pipelines for the entire process which is shown in the __Demo__ section below.

## Demo

![Execution flow](https://github.com/Yongbinkang/ExpFinder/blob/main/images/flow.png)

In this section, we describe the demo of ExpFinder in the __`experimental pipeline.ipynb`__ file. The flow of the demo is presented as follows:

* In the preparation phase, raw data (experts, their documents, and topics) is read and transform to a proper format like dataframes or vectors.
* In the generation phase, we apply statistical models to generate data for our *ExpFinder* algorithm. These includes:
  * Expert-document, document-phrase, document-topic and personalised matrices
  * Expert-document counted vector (counting a number of documents per expert) and Document-expert counted vector (counting a number of experts per document)
* In the training phase, we fitted all generated data into the ExpFinder algorithm with the best hyperparameters based on the empirical experiement. Please refer to our paper for more details. Then, we obtain results from the model which contain reinforced weights of experts and documents given topics.

## Public datasets

Currently, many datasets in our publication(s) below are unavailable. Thus, we decided to re-publish all datasets that we use for experimenting our `ExpFinder` framework. You can find the zip file of all datasets in the `public_data` directory. The zip file contains the following datasets:

1. `LExR` is the Lattes Expertise Retrieval collection for expertise retrieval in academic. More details about the dataset can be found in [The LExR Collection for Expertise Retrieval in Academia](https://dl.acm.org/doi/10.1145/2911451.2914678).
2. `IR-CL-SW` are filtered subsets of DBLP dataset in [Benchmarking domain-specific expert search using workshop program committees](https://dl.acm.org/doi/abs/10.1145/2508497.2508501). The original version of these three datasets can be also found at [this website](http://toinebogers.com/?page_id=240).

## Citation

If you use `ExpFinder` in your research, please cite [ExpFinder: An Ensemble Expert Finding Model Integrating N-gram Vector Space Model and μCO-HITS](https://arxiv.org/abs/2101.06821)

```
@misc{kang2021expfinder,
      title={ExpFinder: An Ensemble Expert Finding Model Integrating $N$-gram Vector Space Model and $\mu$CO-HITS}, 
      author={Yong-Bin Kang and Hung Du and Abdur Rahim Mohammad Forkan and Prem Prakash Jayaraman and Amir Aryani and Timos Sellis},
      year={2021},
      eprint={2101.06821},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

<!--
and [An open-source framework for ExpFinder integrating N-gram vector space model and μCO-HITS](https://doi.org/10.1016/j.simpa.2021.100069)

```
@article{du2021open,
  title={An open-source framework for ExpFinder integrating N-gram vector space model and $\mu$CO-HITS},
  author={Du, Hung and Kang, Yong-Bin},
  journal={Software Impacts},
  pages={100069},
  year={2021},
  publisher={Elsevier}
}
```
-->
