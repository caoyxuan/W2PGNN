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

`pytorch 1.6.0`

`torch-geometric 1.6.3`

`rdkit 2020.09.1.0`

`networkx 2.3`

`scikit-learn 0.20.3`

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
    python Main.py -h

    usage: Main.py [-h][--data-name] [--save-name] [--max-train-num] [--no-cuda] [--missing-ratio] 
    [--split-ratio] [--neg-pos-ratio] [--use-attribute] [--use-embedding] [--embedding-size] 
    [--lazy-subgraph] [--max-nodes-per-hop] [--num-walks] [--multi-subgraph] [--reg-smooth] 
    [--smooth-coef] [--trainable-noise] [--early-stop] [--early-stop-patience] [--learning-rate]

    optional arguments:
      -h, --help                show this help message and exit
      --data-name               str, select the dataset. 
      --save-name               str, the name of saved model. 
      --max-train-num           int, the maximum number of training links.
      --no-cuda                 bool, whether to disables CUDA training.
      --seed                    int, set the random seed.
      --test-ratio              float, the ratio of test links.
      --missing-ratio           float, the ratio of missing links.
      --split-ratio             str, the split rate of train, val and test links
      --neg-pos-ratio           float, the ratio of negative/positive links
      --use-attribute           bool, whether to utilize node attribute. 
      --use-embedding           bool, whether to utilize the information from node2vec node embeddings.
      --embedding-size          int, the embedding size of node2vec
      --lazy-subgraph           bool, whether to use lazy subgraph extraction.
      --max-nodes-per-hop       int, the upper bound the number of nodes per hop when performing Lazy Subgraph Extraction. 
      --num-walks               int, thenumber of walks for each node when performing Lazy Subgraph Extraction. 
      --multi-subgraph          int, the number of subgraphs to extract for each queried nodes
      --reg-smooth              bool, whether to use auxiliary denoising regularization.
      --smooth-coef             float, the coefficient of auxiliary denoising regularization. 
      --trainable-noise         bool, whether to let the Noisy link detection layer trainable.
      --early-stop              bool, whether to use early stopping.
      --early-stop-patience     int, the patience for early stop.
      --learning-rate           float, the learning rate. 
### Demo
To run the node_classification, see example below

`python node_classification/estimate_feasiblity.py --pre_data imdb_facebook --down_data h-index`

To run the graph_classification, see example below

`python graph_classification/estimate_feasiblity.py --pre_data zinc_standard_agent --split_num 2 --split_ids 01 --down_data bbbp`



