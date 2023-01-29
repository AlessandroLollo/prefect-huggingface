"""Block containing credentials to use to authenticate with Hugginface Inference API."""

from prefect.blocks.core import Block
from pydantic import Field, SecretStr

from prefect_huggingface.client import HuggingfaceClient


class HuggingfaceCredentials(Block):
    """
    Block containing credentials to use to authenticate with Huggingface Inference API.

    Attributes:
        access_token: Access token to use to authenticate
            with Huggingface Inference API.

    Examples:
        Load stored Hugginface credentials:
        ```python
        from prefect_huggingface.credentials import HuggingfaceCredentials

        hf_creds = HuggingfaceCredentials.load("BLOCK_NAME")
        ```
    """

    _block_type_name = "Huggingface Credentials"
    _logo_url = "https://avatars.githubusercontent.com/u/25720743?s=200&v=4"  # noqa
    _documentation_url = "https://todo.todo"  # noqa

    access_token: SecretStr = Field(
        ...,
        title="Access token",
        description="Access token to authenticate with Huggingface API.",
    )

    def get_client(self) -> HuggingfaceClient:
        """
        Returns an `HuggingfaceClient` that uses credentials included in the block.

        Returns:
            `HuggingfaceClient` that authenticates with Huggingface Inference API
                using credentials stored in the block.
        """
        return HuggingfaceClient(access_token=self.access_token.get_secret_value())
