# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from functools import cache
from typing import List, Union

import httpx
import tiktoken
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from langchain_aws import ChatBedrockConverse
from openai import AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_message,
    retry_if_exception_type,
    stop_after_delay,
    wait_random,
    wait_random_exponential,
)
from transformers import AutoTokenizer
from vertexai.preview.tokenization import get_tokenizer_for_model

DEFAULT_TOKENIZER_MAPPING = {
    "gemini": "google",
    "vllm": "huggingface",
    "openai": "tiktoken",
    "azure-openai": "tiktoken",
    "aws": "huggingface",
}


class APIConnector:
    def __init__(
        self, api_key: str, api_url: str, api_provider: str, model: str, **kwargs
    ) -> None:
        """
        APIConnector class to unify the API calls for different API providers

        Parameters:
            api_key (`str`): API Key for the API
            api_url (`str`): API URL for the API
            api_provider (`str`): API Provider (openai, gemini, aws, vllm, azure-openai, etc.)
            model (`str`): Model name to use for the API
        """
        self.api_provider = api_provider

        if "tokenizer_type" in kwargs:
            self.tokenizer_type = kwargs["tokenizer_type"]
        else:
            if api_provider in DEFAULT_TOKENIZER_MAPPING:
                self.tokenizer_type = DEFAULT_TOKENIZER_MAPPING[api_provider]
            else:
                raise ValueError(f"Invalid API provider: {api_provider}")

        self.tokenizer_model = (
            kwargs["tokenizer_model"] if "tokenizer_model" in kwargs else model
        )

        if self.tokenizer_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_model, use_fast=True
            )
        elif self.tokenizer_type == "tiktoken":
            self.tokenizer = tiktoken.get_encoding(self.tokenizer_model)
        elif self.tokenizer_type == "google":
            self.tokenizer = get_tokenizer_for_model(self.tokenizer_model)
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

        self.SYSTEM_PROMPT = kwargs.get("system_prompt", "You are a helpful assistant")

        if api_provider == "openai":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
            )
        elif api_provider == "azure-openai":
            self.api = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_url,
                api_version=kwargs["azure_api_version"],
            )
        elif api_provider == "vllm":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
                max_retries=kwargs["max_retries"],
                timeout=kwargs["timeout"],
            )
        elif api_provider == "gemini":
            self.api = genai.Client(
                vertexai=True,
                project=kwargs["project_ID"],
                location=kwargs["location"],
                http_options=types.HttpOptions(timeout=kwargs["http_timeout"] * 1000),
            )

            self.gemini_retry_config = {
                "initial": kwargs["retry_delay"],
                "maximum": kwargs["retry_max_delay"],
                "multiplier": kwargs["retry_multiplier"],
                "timeout": kwargs["retry_timeout"],
            }

        elif api_provider == "aws":
            self.api = ChatBedrockConverse(
                model=model,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
                region_name=kwargs["region"],
                top_p=kwargs["top_p"],
            )

        self.model = model
        self.model_config = {k: v for k, v in kwargs.items()}

    def encode(self, text: str) -> list:
        """
        Encodes the given text using the tokenizer

        Parameters:
            text (`str`): Text to encode

        Returns:
            `dict`: Encoded text
        """
        if self.tokenizer_type == "huggingface":
            return self.tokenizer.encode(text, add_special_tokens=False)
        elif self.tokenizer_type == "tiktoken":
            return self.tokenizer.encode(text)
        elif self.tokenizer_type == "google":
            return self.tokenizer._sentencepiece_adapter._tokenizer.encode(text)
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    def decode(self, tokens: list) -> str:
        """
        Decodes the given tokens using the tokenizer

        Parameters:
            tokens (`list`): Tokens to decode

        Returns:
            `str`: Decoded text
        """
        if self.tokenizer_type == "huggingface":
            return self.tokenizer.decode(tokens)
        elif self.tokenizer_type == "tiktoken":
            return self.tokenizer.decode(tokens)
        elif self.tokenizer_type == "google":
            return self.tokenizer._sentencepiece_adapter._tokenizer.decode(tokens)
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    @cache
    def token_count(self, text: str) -> int:
        """
        Returns the token count of the given text

        Parameters:
            text (`str`): Text to count tokens
            use_cache (`bool`, optional): Use cache for token count. Defaults to True.

        Returns:
            `int`: Token count of the text
        """
        if self.tokenizer_type == "tiktoken":
            return len(self.tokenizer.encode(text))
        elif self.tokenizer_type == "huggingface":
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        elif self.tokenizer_type == "google":
            return self.tokenizer.count_tokens(text).total_tokens
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: Union[str, List[str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        add_default_system_prompt: bool = True,
    ) -> dict:
        """
        Generates a response using the API with the given prompts

        Parameters:
            system_prompt (`str`): System prompt
            user_prompt (`str`): User prompt
            max_tokens (`int`, optional): Maximum tokens to generate. Defaults to 100.
            temperature (`float`, optional): Temperature. Defaults to 0.0 (Greedy Sampling).
            top_p (`float`, optional): Top-p. Defaults to 1.0.

        Returns:
            `dict`: Response from the API that includes the response, prompt tokens count, completion tokens count, total tokens count, and stopping reason
        """
        if add_default_system_prompt and self.SYSTEM_PROMPT != "":
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        elif system_prompt != "":
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = []
        if isinstance(user_prompt, list):
            for prompt in user_prompt:
                messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})
        if (
            self.api_provider == "openai"
            or self.api_provider == "vllm"
            or self.api_provider == "azure-openai"
        ):

            @retry(
                reraise=True,
                wait=wait_random(1, 20),
                retry=retry_if_exception_type(RateLimitError),
                stop=stop_after_delay(300),
            )
            async def generate_content():
                params = {"model": self.model, "messages": messages, "seed": 43}
                if self.model_config.get("openai_thinking_model", False):
                    params["max_completion_tokens"] = max_tokens
                else:
                    params["max_tokens"] = max_tokens
                    params["temperature"] = temperature
                    params["top_p"] = top_p

                completion = await self.api.chat.completions.create(**params)
                return completion

            completion = await generate_content()

            output = {
                "response": completion.choices[0].message.content,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
                "finish_reason": completion.choices[0].finish_reason,
                "cached_tokens": completion.usage.prompt_tokens_details.cached_tokens
                if completion.usage.prompt_tokens_details
                else None,
            }
            if self.model_config.get("openai_thinking_model", False):
                output["reasoning_tokens"] = (
                    completion.usage.completion_tokens_details.reasoning_tokens
                )
            return output

        elif self.api_provider == "gemini":

            @retry(
                reraise=True,
                wait=wait_random_exponential(
                    multiplier=self.gemini_retry_config["multiplier"],
                    max=self.gemini_retry_config["maximum"],
                ),
                stop=stop_after_delay(self.gemini_retry_config["timeout"]),
                retry=retry_if_exception_type(
                    (ClientError, httpx.ConnectTimeout, httpx.TimeoutException)
                ),
            )
            async def generate_content():
                completion = await self.api.aio.models.generate_content(
                    model=self.model,
                    contents=messages[-1]["content"],
                    config=types.GenerateContentConfig(
                        system_instruction=self.SYSTEM_PROMPT,
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=self.model_config["thinking_budget"]
                        )
                        if "thinking_budget" in self.model_config
                        else None,
                        seed=43,
                    ),
                )
                return completion

            completion = await generate_content()

            return {
                "response": completion.text if completion.text is not None else "",
                "prompt_tokens": completion.usage_metadata.prompt_token_count,
                "completion_tokens": completion.usage_metadata.candidates_token_count,
                "total_tokens": completion.usage_metadata.total_token_count,
                "finish_reason": completion.candidates[0].finish_reason,
                "cached_tokens": completion.usage_metadata.cached_content_token_count
                if completion.usage_metadata.cached_content_token_count is not None
                else None,
                "reasoning_tokens": completion.usage_metadata.thoughts_token_count
                if completion.usage_metadata.thoughts_token_count is not None
                else None,
            }
        elif self.api_provider == "aws":

            @retry(
                reraise=True,
                wait=wait_random(5, 20),
                retry=retry_if_exception_message(match=r".*ThrottlingException.*"),
            )
            async def generate_content():
                completion = await self.api.ainvoke(messages)
                return completion

            completion = await generate_content()

            return {
                "response": completion.content,
                "prompt_tokens": completion.usage_metadata["input_tokens"],
                "completion_tokens": completion.usage_metadata["output_tokens"],
                "total_tokens": completion.usage_metadata["total_tokens"],
                "finish_reason": completion.response_metadata["stopReason"],
            }
        else:
            raise ValueError(f"Invalid API provider: {self.api_provider}")
