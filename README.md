# historical_scent
Thesis Repository where I utilize topic models to analyze the effectiveness of topic models on short historical texts regarding scent from 1600-1980 in English and Dutch. This repository is an addition to the research I did for my masters thesis. I hope this repository can be used as a framework for applying topic modeling to similar datasets.

## Data

For this project I used the data extracted by the Odeuropa project. The odeuropa project extracted data from corpora where Scent is mentioned, creating a database of Scentences where smell is mentioned in texts from 1600-1980. This data can be downloaded here: https://colab.research.google.com/drive/1spHM3qFtkuQDoIzQ3ZA94BILvYgGm_Ud
We extracted the English and the Dutch subset of the data that can be found over here:
https://drive.google.com/drive/folders/1RrIRFbWzuxsqF9N-c3qCZRXDr0YeB_5d?usp=drive_link

## OCTIS

For the implementation of the models and the preprocessing of the dataset, we utilized the OCTIS framework. LINK TO FRAMEWORK: https://github.com/MIND-Lab/OCTIS

## MODELS

For this project, we tested 6 different models: LDA, CTM, NMF, BERTopic (pretrained / not pretrained), Dynamic BERTopic. On 4 different datasets. (Dutch and English with focus on Flowers, and the modernized version of the datasets). For the execution of the models we took inspiration from Grootendorst's project listed over here: https://github.com/MaartenGr/BERTopic . We tried to keep our experiments as similar as possible to proof generalizability, but a lot of adjustments needed to be made based on data characteristics and optimizaton methods.

## EVALUATION

For the evaluation of this model, we used OCTIS to compute the Topic Coherence and Topic Diversity. The actual topics can be found in Appendix A of the research paper and computed by running the model. The generated models are used for futher human evaluation.

## Framework usage guide.
### Dataset creation:
First of all, the 4 used datasets in our experiments are listed in the 'used_datasets' folder. To create the datasets, you can download the data from the drive link listed in the data section above and run the en_full_pre.ipynb, nl_full_pre.ipynb, pre_mod_create.ipynb notebooks in the data_Prep models. But since the models are already in used_datasets, this is not necessary. 
### Model implementation and evaluation
The next step is to run the Data_Prep.ipynb in the data_Prep folder. This creates the data form required for the OCTIS framework, including vocabularies, for both the TMs and the DTMs. The notebook uses the scripts listed in the same folder. For the final implementation of the models you can run the models in the TM_implementation folder, that utilize the evaluaton3 and evaluation_vis scripts. For each model you can run the experiment to compute topic coherence and topic diversity, that are then stored in json files and run the model that generates the topics, used for topic evaluation, that are also stored as json files.

