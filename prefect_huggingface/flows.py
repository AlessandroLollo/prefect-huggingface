"""This is an example flows module"""
from prefect import flow

from prefect_huggingface.credentials import HuggingfaceBlock
from prefect_huggingface.tasks import (
    goodbye_prefect_huggingface,
    hello_prefect_huggingface,
)


@flow
def hello_and_goodbye():
    """
    Sample flow that says hello and goodbye!
    """
    HuggingfaceBlock.seed_value_for_example()
    block = HuggingfaceBlock.load("sample-block")

    print(hello_prefect_huggingface())
    print(f"The block's value: {block.value}")
    print(goodbye_prefect_huggingface())
    return "Done"


if __name__ == "__main__":
    hello_and_goodbye()
