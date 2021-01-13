# ExpFinder: An Ensemble Expert Finding Model Integrating N-gram Vector Space Model and μCO-HITS

## Introduction
<p align="justify">
Finding an expert plays a crucial role in driving successful collaborations and speeding up high-quality research development and innovations. However, the rapid growth of scientific publications and digital expertise data makes identifying the right experts a challenging problem. Existing approaches for finding experts given a topic can be categorised into information retrieval techniques based on vector space models, document language models, and graph-based models. In this paper, we propose <i>ExpFinder</i>, a new ensemble model for expert finding, that integrates a novel <i>N</i>-gram vector space model, denoted as <i>n</i>VSM, and a graph-based model, denoted as <i>μCO-HITS</i>, that is a proposed variation of the CO-HITS algorithm. The key of <i>n</i>VSM is to exploit recent inverse document frequency weighting method for <i>N</i>-gram words, and <i>ExpFinder</i> incorporates <i>n</i>VSM into <i>μCO-HITS</i> to achieve expert finding. We comprehensively evaluate <i>ExpFinder</i> on four different datasets from the academic domains in comparison with six different expert finding models. The evaluation results show that <i>ExpFinder</i> is an highly effective model for expert finding, substantially outperforming all the compared models in 19% to 160.2%.
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
3. Download the SciBert model into the `model/` folder as mentioned in [this](https://github.com/Yongbinkang/ExpFinder/tree/main/model).

## Notes
1. The `data` folder stores all required data for the algorithm.
2. Pipelines for data generation and algorithm are presented in the `experimental pipline.ipynb` file. We recommend to clone the repository and run the file locally for more visualable view.
3. Details of the source code are stored in the `src` folder with clear documentation for each functions.

## Citing
