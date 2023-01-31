"""Exceptions to be raised in case of failures with Huggingface API."""


class HuggingfaceInferenceConfiguration(Exception):
    """Exception to be raised in case of configuration error."""


class HuggingfaceAPIFailure(Exception):
    """Excepion to be raise in case of failure with Huggingface API."""
