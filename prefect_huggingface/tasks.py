"""Tasks to interact with Huggingface Inference API."""

from typing import Dict, Optional

from prefect import task

from prefect_huggingface.credentials import HuggingfaceCredentials


@task
def get_inference_result(
    credentials: HuggingfaceCredentials,
    model_id: str,
    inputs: str,
    options: Optional[Dict],
    parameters: Optional[Dict],
) -> Dict:
    """
    TODO
    """
    hf_client = credentials.get_client()
    return hf_client.get_inference_result(
        model_id=model_id, inputs=inputs, options=options, parameters=parameters
    )
