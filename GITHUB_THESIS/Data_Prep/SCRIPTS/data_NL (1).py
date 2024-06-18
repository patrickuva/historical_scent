import re
import nltk
import string
import pandas as pd

from typing import List, Tuple, Union
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing

nltk.download("punkt")


class DataLoader:
    """Prepare and load custom data using OCTIS

    Arguments:
        dataset: The name of the dataset, default options:
                    * NL

    Usage:

    **NL** - Unprocessed

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="NL").prepare_docs(save="NL.txt").preprocess_octis(output_folder="NL")
    ```

    **20 Newsgroups** - Unprocessed

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="20news").prepare_docs(save="20news.txt").preprocess_octis(output_folder="20news")
    ```

    **Custom Data**

    Whenever you want to use a custom dataset (list of strings), make sure to use the loader like this:

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs).preprocess_octis(output_folder="my_docs")
    ```
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.docs = None
        self.timestamps = None
        self.octis_docs = None
        self.doc_path = None

    def load_docs(
        self, save: bool = False, docs: List[str] = None
    ) -> Tuple[List[str], Union[List[str], None]]:
        """Load in the documents

        ```python
        dataloader = DataLoader(dataset="NL")
        docs, timestamps = dataloader.load_docs()
        ```
        """
        if docs is not None:
            return self.docs, None

        if self.dataset == "NL":
            self.docs, self.timestamps = self._NL()
        elif self.dataset == "NL_dtm":
            self.docs, self.timestamps = self._NL_dtm()
        elif self.dataset == "un_dtm":
            self.docs, self.timestamps = self._un_dtm()
        elif self.dataset == "20news":
            self.docs, self.timestamps = self._20news()

        if save:
            self._save(self.docs, save)

        return self.docs, self.timestamps

    def load_octis(self, custom: bool = False) -> Dataset:
        """Get dataset from OCTIS

        Arguments:
            custom: Whether a custom dataset is used or one retrieved from
                    https://github.com/MIND-Lab/OCTIS#available-datasets

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="20news")
        data = dataloader.load_octis(custom=True)
        ```
        """
        data = Dataset()

        if custom:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)

        self.octis_docs = data
        return self.octis_docs

    def prepare_docs(self, save: bool = False, docs: List[str] = None):
        """Prepare documents

        Arguments:
            save: The path to save the model to, make sure it ends in .json
            docs: The documents you want to preprocess in OCTIS

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        ```
        """
        self.load_docs(save, docs)
        return self

    def preprocess_octis(
        self,
        preprocessor: Preprocessing = None,
        documents_path: str = None,
        output_folder: str = "docs",
    ):
        """Preprocess the data using OCTIS

        Arguments:
            preprocessor: Custom OCTIS preprocessor
            documents_path: Path to the .txt file
            output_folder: Path to where you want to save the preprocessed data

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        dataloader.preprocess_octis(output_folder="my_docs")
        ```

        If you want to use your custom preprocessor:

        ```python
        from evaluation import DataLoader
        from octis.preprocessing.preprocessing import Preprocessing

        preprocessor = Preprocessing(lowercase=False,
                                remove_punctuation=False,
                                punctuation=string.punctuation,
                                remove_numbers=False,
                                lemmatize=False,
                                language='english',
                                split=False,
                                verbose=True,
                                save_original_indexes=True,
                                remove_stopwords_spacy=False)

        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        dataloader.preprocess_octis(preprocessor=preprocessor, output_folder="my_docs")
        ```
        """
        if preprocessor is None:
            preprocessor = Preprocessing(
                lowercase=False,
                remove_punctuation=False,
                punctuation=string.punctuation,
                remove_numbers=False,
                lemmatize=False,
                language="english",
                split=False,
                verbose=True,
                save_original_indexes=True,
                remove_stopwords_spacy=False,
            )
        if not documents_path:
            documents_path = self.doc_path
        dataset = preprocessor.preprocess_dataset(documents_path=documents_path)
        dataset.save(output_folder)

    def _NL(self) -> Tuple[List[str], List[str]]:
        """Prepare the NL dataset"""
        NL = pd.read_csv("full_NL.csv")
        
        # Drop rows where 'modern_sentences' is NaN
        NL = NL.dropna(subset=['pre_text'])
        # Drop rows where 'modern_sentences' is NaN
        NL = NL.dropna(subset=['year'])
        
        
        timestamps = NL.year.to_list()
        docs = NL.pre_text.to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        timestamps = [
            timestamp for timestamp, doc in zip(timestamps, docs) if len(doc) > 2
        ]
        return docs, timestamps

    def _NL_dtm(self) -> Tuple[List[str], List[str]]:
        """Prepare the NL dataset including timestamps"""
        NL = pd.read_csv(
            "full_NL.csv"
        )
        
        # Drop rows where 'modern_sentences' is NaN
        NL = NL.dropna(subset=['pre_text'])
        # Drop rows where 'modern_sentences' is NaN
        NL = NL.dropna(subset=['year'])
        
        
        #NL["text"] = NL.apply(lambda row: create_paragraphs(row.pre_text), 1)
        NL = NL.explode("pre_text").sort_values("year")
        #dataset = dataset.loc[dataset.year > 2005, :]
        docs = NL.pre_text.tolist()
        timestamps = NL.year.tolist()
        return docs, timestamps

    def _save(self, docs: List[str], save: str):
        """Save the documents"""
        with open(save, mode="wt", encoding="utf-8") as myfile:
            myfile.write("\n".join(docs))

        self.doc_path = save
