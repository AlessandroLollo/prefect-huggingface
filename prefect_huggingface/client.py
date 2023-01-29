"""TODO"""

import json
from typing import Dict

from requests.sessions import Session


class HuggingfaceClient:
    """
    TODO
    """

    def __init__(self, access_token: str) -> None:
        """
        TODO
        """
        self.access_token = access_token
        self.inferece_api_base_url = "https://api-inference.huggingface.co"

    def __get_model_url(self, model_id: str) -> str:
        """
        TODO
        """
        return f"{self.inferece_api_base_url}/models/{model_id}"

    def __get_session(self) -> Session:
        """
        TODO
        """
        session = Session()
        session.headers = {"Authorization": f"Bearer {self.access_token}"}

        return session

    def get_inference_result(
        self, model_id: str, inputs: str, options: Dict = None, parameters: Dict = None
    ) -> Dict:
        """
        TODO
        """
        url = self.__get_model_url(model_id=model_id)
        session = self.__get_session()
        data = {"inputs": inputs}
        if options:
            data["options"] = options
        if parameters:
            data["parameters"] = parameters

        with session.post(url=url, data=json.dumps(data)) as response:
            if response.status_code != 200:
                raise Exception("Error!")
            else:
                return json.loads(response.content.decode("utf-8"))
