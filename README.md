# historical_scent
Thesis Repository where I utilize topic models to analyze the effectiveness of topic models on short historical texts regarding scent from 1600-1980 in English and Dutch.

## Data

For this project I used the data extracted by the Odeuropa project. The odeuropa project extracted data from corpora where Scent is mentioned, creating a database of Scentences where smell is mentioned in texts from 1600-1980. This data can be downloaded here:
-- Drive Link -- 

## OCTIS

For the optimization of the models and the preprocessing of the dataset, we utilized the OCTIS framework. LINK TO FRAMEWORK:

## MODELS

For this project, we tested 9 different models: LDA, Neural LDA, CTM, ETM, NMF, LDASeq, BERTopic (pretrained / not pretrained), Dynamic BERTopic. On 2 different datasets. (Dutch and English with focus on Flowers).

## EVALUATION

For the evaluation of this model, we used OCTIS to compute the Topic Coherence and Topic Diversity. The actual topics can be found in -- FILE -- and can be used for futher human evaluation.
