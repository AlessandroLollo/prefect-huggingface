"""This is an example blocks module"""

from prefect.blocks.core import Block
from pydantic import Field, SecretStr


class HuggingfaceCredentials(Block):

    _block_type_name = "Huggingface Credentials"
    _logo_url = "https://todo.todo"  # noqa

    access_token: SecretStr = Field(
        ...,
        title="Access token",
        description="Access token to authenticate with Huggingface API."
    )

    def get_client(self) -> HuggingfaceClient:
        """
        TODO
        """
        pass