'''Library for comparing sentences using sentence_transformers'''
import re
import csv
import json
import time
import scipy
import pickle
import logging
from typing import List, AnyStr
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SentenceDataset():
    def __init__(self, data):
        self.data = [{"id": a[0], "text":a[1]} for a in enumerate(data)]
        self.len = len(self.data)
    #end def

    def _read_from_file(self) -> List[dict]:
        with open(self.file_path, 'r') as f:
            return [{"id" : row_id, "text" : row.replace("\n", "")} for row_id, row in enumerate(f.readlines())]
    #end def

    def get_documents_by_id(self, doc_ids: List[int]) -> List[str]:
        return [self.data[doc_id]["text"] for doc_id in doc_ids]

    def get_documents(self, n: int = -1) -> dict:
        for i, row in enumerate(self.data):
            if i == n:
                break
            yield row

    def __len__(self):
        return self.len
    #end def

    def __str__(self):
        return f"I am a collection of {self.len} sentences"
    #end def
#end class


class SentenceSimilarity():
    def __init__(self, dataset: list, model: SentenceTransformer = None, n_docs: int = -1):
        self.dataset = SentenceDataset(dataset)
        self.model = model if model else SentenceTransformer("bert-base-nli-stsb-mean-tokens")

        self.sentences = []
        self.doc_id_to_sentence_ids = {}

        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

        for d in self.dataset.get_documents(n=n_docs):
            doc_id = d.get('id')
            text = d.get('text', None)

            sentence_ids = []
            if text:
                sentences = re.split(self.sentence_pattern, text)
                for s in sentences:
                    sentence_ids.append(len(self.sentences))
                    self.sentences.append(s)

            # Map from document to all its sentences (One-to-Many)
            self.doc_id_to_sentence_ids[doc_id] = sentence_ids
        #end for

        logger.debug(f"doc_to_sentence_ids: {self.doc_id_to_sentence_ids}")

        # Map from sentence to the document it came from (Many-to-One)
        self.sentence_id_to_doc_id =  {}
        for doc_id, sentence_ids in self.doc_id_to_sentence_ids.items():
            for s_id in sentence_ids:
                self.sentence_id_to_doc_id[s_id] = doc_id

        logger.debug(f"sentence_id_to_doc_id: {self.sentence_id_to_doc_id}")
        # Embedd extracted sentences using SentenceTransformer model.
        start = time.time()
        self.embedded_sentences = self.model.encode(self.sentences)
        logger.info(f"It took {round(time.time()-start, 3)} s to embedd {len(self.sentences)} sentences.")

        # temp  = []
        # for doc_id, sentence_ids in self.doc_id_to_sentence_ids.items():
        #     logger.debug((doc_id, sentence_ids))
        #     temp.append(f"document: {self.dataset.get_documents_by_id([doc_id])} - sentences: {[self.sentences[sid] for sid in sentence_ids]}")
        # logger.debug("\n\n".join(temp))
    #end def


    def get_most_similar(self, query: AnyStr, threshold: float = 1, limit: int = 10) -> List[int]:
        query_sentences = re.split(self.sentence_pattern, query)
        query_embeddings = self.model.encode(query_sentences)

        logger.info(f"Extracted {len(query_sentences)} sentences from query")
        logger.debug(f"Sentences: {' -- '.join(query_sentences)}")

        # Calculate cosine distance between requested sentences and all sentences
        cosine_dist = scipy.spatial.distance.cdist(query_embeddings, self.embedded_sentences, "cosine")
        # Extract column values where distance is below threshold
        below_threshold = cosine_dist < threshold
        doc_ids, matched_column_ids = np.where(below_threshold)

        # Extract x (input sentence id), y (dataset sentence id) and distance between these.
        x_y_dist = []
        for x,y in zip(doc_ids, matched_column_ids):
            x_y_dist.append([x,y,cosine_dist[x][y]])

        # Sort list based on distance and remove duplicates, keeping the one with lowest distance.
        sorted_x_y_dist = sorted(x_y_dist, key=lambda x: x[2])
        sorted_sentence_ids = [doc[1] for doc in sorted_x_y_dist]
        sorted_doc_ids = [self.sentence_id_to_doc_id[sent_id] for sent_id in sorted_sentence_ids]

        logger.info(f"Distance for top documents: {[round(x[2],3) for x in sorted_x_y_dist[:limit]]}")
        return self.dataset.get_documents_by_id(list(dict.fromkeys(sorted_doc_ids).keys())[:limit])
    #end def
