from prefect import flow

from prefect_huggingface.tasks import (
    goodbye_prefect_huggingface,
    hello_prefect_huggingface,
)


def test_hello_prefect_huggingface():
    @flow
    def test_flow():
        return hello_prefect_huggingface()

    result = test_flow()
    assert result == "Hello, prefect-huggingface!"


def goodbye_hello_prefect_huggingface():
    @flow
    def test_flow():
        return goodbye_prefect_huggingface()

    result = test_flow()
    assert result == "Goodbye, prefect-huggingface!"
