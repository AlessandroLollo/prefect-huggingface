import logging
from typing import Dict

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from prefect_huggingface.managers.text_classification_models import SentimentAnalysisModel
from prefect_huggingface.settings import (
    SENTIMENT_ANALYSIS_TASK,
    SENTIMENT_ANALYSIS_PATH,
)

logger = logging.getLogger(__name__)


class InstantiateTransformerComponents:

    MAPPING: Dict = {
        'cpu': -1,
        'cuda': 0,
    }

    def __init__(self, model_path: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = model

    def retrieve_pipeline(self, task: str, device: int = 'cpu', use_fast_tokenizer: bool = True):
        return pipeline(
            task=task,
            model=self.model_path,
            tokenizer=self.model_path,
            device=self.MAPPING[device],
            use_fast=use_fast_tokenizer
        )


class LoadTransformersComponents:
    """
    This class instantiates the desired model for the specified task
    Here basically we are going to map each single task with a specific model path
    """
    MAPPING = {
        SENTIMENT_ANALYSIS_TASK: InstantiateTransformerComponents(
            model_path=SENTIMENT_ANALYSIS_PATH, tokenizer=AutoTokenizer, model=AutoModelForSequenceClassification
        ),
    }


class LoadTransformerPipeline:
    """
    This class instantiates the desired model for the specified task
    Here basically we are going to map each single task with a specific model path

    """
    def __init__(self, task_pipeline: str, my_pipeline: Pipeline):
        self.task_pipeline = task_pipeline
        self.my_pipeline = my_pipeline

    def get_model(self):
        MAPPING = {
            SENTIMENT_ANALYSIS_TASK: SentimentAnalysisModel(self.my_pipeline)
        }
        return MAPPING[self.task_pipeline]


class ModelManager:

    @classmethod
    def get_model(cls, transformer_task: str, device: str, use_fast_tokenizer: bool = True):
        if isinstance(transformer_task, str):
            if transformer_task not in list(LoadTransformersComponents.MAPPING.keys()) or transformer_task is None:
                possible_tasks = ", ".join([elem for elem in LoadTransformersComponents.MAPPING.keys()])
                raise ValueError(
                    f"Please Provide a valid Transformer Task among the following ones: {possible_tasks}",
                )

            transformer_loader = LoadTransformersComponents.MAPPING[transformer_task]
            pipeline = transformer_loader.retrieve_pipeline(
                task=transformer_task,
                device=device,
                use_fast_tokenizer=use_fast_tokenizer
            )
            logger.info(f'We have loaded the following Pipeline object: {pipeline.__class__.__name__}')
            transformer_service = LoadTransformerPipeline(transformer_task, pipeline).get_model()

        else:
            raise ValueError(f"Bad model init parameter: {transformer_task}")

        return transformer_service
