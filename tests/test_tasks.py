import pytest
import responses
from prefect import flow
from pydantic import SecretStr

from prefect_huggingface.credentials import HuggingfaceCredentials
from prefect_huggingface.exceptions import HuggingfaceAPIFailure
from prefect_huggingface.tasks import get_inference_result


@responses.activate
def test_get_inference_result_fails():
    access_token = "token"
    model_id = "model"
    inputs = "inputs"
    options = {"options": "opts"}
    parameters = {"parameters": "params"}

    @flow
    def test_flow():
        return get_inference_result(
            credentials=HuggingfaceCredentials(access_token=SecretStr(access_token)),
            inference_endpoint_url=None,
            model_id=model_id,
            inputs=inputs,
            options=options,
            parameters=parameters,
        )

    responses.add(
        method=responses.POST,
        url=f"https://api-inference.huggingface.co/models/{model_id}",
        status=123,
    )

    msg_match = (
        "There was an error while retrieving result from Huggingface API."  # noqa
    )

    with pytest.raises(HuggingfaceAPIFailure, match=msg_match):
        test_flow()


@responses.activate
def test_get_inference_result_succeed():
    access_token = "token"
    model_id = "model"
    inputs = "inputs"
    options = {"options": "opts"}
    parameters = {"parameters": "params"}

    expected_result = {"result": "ok"}

    @flow
    def test_flow():
        return get_inference_result(
            credentials=HuggingfaceCredentials(access_token=SecretStr(access_token)),
            inference_endpoint_url=None,
            model_id=model_id,
            inputs=inputs,
            options=options,
            parameters=parameters,
        )

    responses.add(
        method=responses.POST,
        url=f"https://api-inference.huggingface.co/models/{model_id}",
        status=200,
        json=expected_result,
    )

    result = test_flow()
    assert result == expected_result
