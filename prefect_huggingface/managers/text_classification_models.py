import torch

from typing import List
from abc import ABC, abstractmethod

from transformers import (
    TextClassificationPipeline,
    pipelines,
)

import logging
logger = logging.getLogger(__name__)


class BaseSentimentAnalysisModelService(ABC):

    @abstractmethod
    def get_sentiment(self, texts: List) -> List[List]:

        """
        This method returns the match between two generic texts.
        :texts: List of strings.
        :return: a List containing list of probabilities
        """
        raise NotImplementedError


class SentimentAnalysisModel(BaseSentimentAnalysisModelService):

    def __init__(self, pipeline: TextClassificationPipeline):
        if pipeline is None or pipeline.__class__ != pipelines.TextClassificationPipeline:
            raise ValueError("Please provide a valid Sentiment Analysis pipeline object")
        self.pipeline = pipeline

        device_available = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(
            f'The device we are using id: {self.pipeline.device} and the one available now is: {device_available}'
        )

    def get_sentiment(self, texts):
        return self.pipeline(texts)
