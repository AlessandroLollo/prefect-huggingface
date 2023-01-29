from pydantic import SecretStr

from prefect_huggingface.client import HuggingfaceClient
from prefect_huggingface.credentials import HuggingfaceCredentials


def test_credentials():
    cr = HuggingfaceCredentials(access_token=SecretStr("token"))
    assert cr.access_token.get_secret_value() == "token"


def test_get_client():
    access_token = "token"
    cr = HuggingfaceCredentials(access_token=SecretStr(access_token))
    expected_client = HuggingfaceClient(access_token=access_token)

    assert expected_client.access_token == cr.get_client().access_token
    assert (
        expected_client.inferece_api_base_url == cr.get_client().inferece_api_base_url
    )
