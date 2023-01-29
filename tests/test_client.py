import pytest
import responses

from prefect_huggingface.client import HuggingfaceClient


def test_client_construction():
    access_token = "token"
    c = HuggingfaceClient(access_token=access_token)
    assert c.access_token == access_token


@responses.activate
def test_get_inference_result_raises():
    model_id = "model"
    inputs = "test"
    access_token = "token"

    c = HuggingfaceClient(access_token=access_token)

    msg_match = "Error!"

    responses.add(
        method=responses.POST,
        url=f"https://api-inference.huggingface.co/models/{model_id}",
        status=123,
    )

    with pytest.raises(Exception, match=msg_match):
        c.get_inference_result(model_id=model_id, inputs=inputs)


@responses.activate
def test_get_inference_result_with_inputs():
    model_id = "model"
    inputs = "test"
    access_token = "token"

    expected_result = {"result": "ok"}

    c = HuggingfaceClient(access_token=access_token)

    responses.add(
        method=responses.POST,
        url=f"https://api-inference.huggingface.co/models/{model_id}",
        status=200,
        json={"result": "ok"},
    )

    result = c.get_inference_result(model_id=model_id, inputs=inputs)

    assert result == expected_result


@responses.activate
def test_get_inference_result_with_inputs_and_options():
    model_id = "model"
    inputs = "test"
    options = {"options": "opt"}
    access_token = "token"

    expected_result = {"result": "ok"}

    c = HuggingfaceClient(access_token=access_token)

    responses.add(
        method=responses.POST,
        url=f"https://api-inference.huggingface.co/models/{model_id}",
        status=200,
        json={"result": "ok"},
    )

    result = c.get_inference_result(model_id=model_id, inputs=inputs, options=options)

    assert result == expected_result


@responses.activate
def test_get_inference_result_with_inputs_and_parameters():
    model_id = "model"
    inputs = "test"
    parameters = {"parameters": "param"}
    access_token = "token"

    expected_result = {"result": "ok"}

    c = HuggingfaceClient(access_token=access_token)

    responses.add(
        method=responses.POST,
        url=f"https://api-inference.huggingface.co/models/{model_id}",
        status=200,
        json={"result": "ok"},
    )

    result = c.get_inference_result(
        model_id=model_id, inputs=inputs, parameters=parameters
    )

    assert result == expected_result
