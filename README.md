# When to Pre-Train Graph Neural Networks? From Data Generation Perspective!
### About
 This project is the implementation of paper “When to Pre-Train Graph Neural Networks? From Data Generation Perspective!" accepted by KDD2023. 
![image](https://github.com/caoyxuan/W2PGNN/blob/main/framework.png)

### Abstract
In recent years, graph pre-training has gained significant attention, focusing on acquiring transferable knowledge from unlabeled graph data to improve downstream performance. 
Despite these recent endeavors, the problem of negative transfer remains a major concern when utilizing graph pre-trained models to downstream tasks. Previous studies made great efforts on the issue of *what to pre-train* and *how to pre-train* by designing a variety of graph pre-training and fine-tuning strategies. However, there are cases where even the most advanced "pre-train and fine-tune" paradigms fail to yield distinct benefits.
This paper introduces a generic framework W2PGNN to answer the crucial question of *when to pre-train* (*i.e.*, in what situations could we take advantage of graph pre-training) before performing effortful pre-training or fine-tuning. We start from a new perspective to explore the complex generative mechanisms from the pre-training data to downstream data. In particular, W2PGNN first fits the pre-training data into graphon bases, each element of graphon basis (*i.e.*, a graphon) identifies a fundamental transferable pattern shared by a collection of pre-training graphs. All convex combinations of graphon bases give rise to a generator space, from which graphs generated form the solution space for those downstream data that can benefit from pre-training. In this manner, the feasibility of pre-training can be quantified as the generation probability of the downstream data from any generator in the generator space. W2PGNN offers three broad applications: providing the application scope of graph pre-trained models, quantifying the feasibility of pre-training, and assistance in selecting pre-training data to enhance downstream performance. We provide a theoretically sound solution for the first application and extensive empirical justifications for the latter two applications.

### Setup
The script has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

`pytorch==1.6.0`

`torch-geometric==1.6.3`

`rdkit==2020.09.1.0`

`networkx==2.3`

`scikit-learn==0.20.3`

You can install the dependency packages with the following command:

`pip install -r requirements.txt`

### File Folders
* node classification: contains the code of estimating graph pre-training feasibility and pre-traing&fine-tuning data for graph classification

* graph classification: contains the code of estimating graph pre-training feasibility and pre-traing&fine-tuning data for node classification

* utils:contains the code of models.

### Dataset
For the graph classiciation pre-training and downstream dataset, download from pre-training data, unzip it, and put it under graph_classification/data/dataset/

For the node classiciation pre-training dataset, the original datasets are stored in data.bin. And the datasets can be download through pre-training data, unzip it, and put it under node_classification/data/dataset/

### Usage: How to run the code
    python estimate_feasiblity.py -h

    usage: estimate_feasiblity.py [-h][--pre-data][--down-data][--pre-path][--down-path][--save-path]
    [--device][--seed][--method][--func][--r][--domain-num][--epoch][--learning-rate][--weight-decay]
    optional arguments:
      -h, --help                show this help message and exit
      --pre-data                str, pretrain data name. 
      --down-data               str, downstream data name. 
      --pre-path                str, file path to load pre-training data.
      --down-path               str, file path to load downstream data.
      --save-path               str, file path to save graphon.
      --file-path               str, Path to save raw data.
      --device                  int, which gpu to use if any (default: 0).
      --seed                    int, set the random seed.
      --method                  str, method to estimate graphon.
      --func                    bool, whether use graphon or step function.
      --r                       str, the resolution of graphon.
      --domain-num              int, the number of domains of pre-training data
      --epochs                  int,  number of trails to fit the final graphon
      --learning-rate           float, the learning rate
      --weight-decay            float, weight decay
      --variant                 str, variant of feasibility
      
### Demo
To run the node_classification, see example below

`python node_classification/estimate_feasiblity.py --pre-datasets imdb facebook --down-data h-index`

To run the graph_classification, see example below

`python graph_classification/estimate_feasiblity.py --pre-data zinc_standard_agent01 --down-data bbbp`



