# historical_scent
Thesis Repository where I utilize topic models to analyze the effectiveness of topic models on short historical texts regarding scent from 1600-1980 in English and Dutch. This repository is an addition to the research I did for my masters thesis. I hope this repository can be used as a framework for applying topic modeling to similar datasets.

## Data

For this project I used the data extracted by the Odeuropa project. The odeuropa project extracted data from corpora where Scent is mentioned, creating a database of Scentences where smell is mentioned in texts from 1600-1980. This data can be downloaded here:
-- Drive Link -- 

## OCTIS

For the implementation of the models and the preprocessing of the dataset, we utilized the OCTIS framework. LINK TO FRAMEWORK:

## MODELS

For this project, we tested 6 different models: LDA, CTM, NMF, BERTopic (pretrained / not pretrained), Dynamic BERTopic. On 4 different datasets. (Dutch and English with focus on Flowers, and the modernized version of the datasets).

## EVALUATION

For the evaluation of this model, we used OCTIS to compute the Topic Coherence and Topic Diversity. The actual topics can be found in -- FILE -- and can be used for futher human evaluation.
