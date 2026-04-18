# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import threading
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import httpx
import ray
import ray.actor

from verl.tools.utils.search_r1_like_utils import (
    async_perform_single_search_batch,
    perform_single_search_batch,
)
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter as a detached actor so it survives the
        death of any individual SearchExecutionWorker that happens to create it
        first. Without lifetime="detached", Ray makes the first creator the owner;
        when that creator dies the rate-limiter is GCed and every other worker's
        acquire/release raises ActorDiedError, cascading into worker SIGABRTs.
        """
        return TokenBucketWorker.options(
            name="rate-limiter",
            get_if_exists=True,
            lifetime="detached",
            namespace="verl-search-tool",
        ).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting.

        Robust against rate-limiter death: if acquire/release fails we fall back
        to running the function without global rate limiting rather than letting
        the exception propagate up and SIGABRT the worker.
        """
        if self.rate_limit_worker:
            acquired = False
            try:
                ray.get(self.rate_limit_worker.acquire.remote())
                acquired = True
            except Exception as e:
                logger.warning(
                    f"Rate limiter acquire failed ({e}); proceeding without global rate limit"
                )
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                logger.warning(f"Error when executing search: {e}")
                return None
            finally:
                if acquired:
                    try:
                        ray.get(self.rate_limit_worker.release.remote())
                    except Exception as e:
                        logger.warning(f"Rate limiter release failed: {e}")
        else:
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                logger.warning(f"Error when executing search: {e}")
                return None


def init_search_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SearchTool(BaseTool):
    """Search tool for retrieving information using external retrieval services.

    This tool provides search functionality with rate limiting and concurrent execution
    support through Ray. It integrates with external retrieval services to perform
    semantic search operations.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration (kept for the Ray path)
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        # Async HTTP path (default). Bypasses the Ray actor funnel and talks
        # to the retriever directly from the agent loop's asyncio event loop
        # using httpx.AsyncClient. Concurrency is bounded in-process by an
        # asyncio.Semaphore (max_concurrent_requests).
        self.use_async_http = config.get("use_async_http", True)
        self.max_concurrent_requests = config.get("max_concurrent_requests", 256)
        self.http_max_connections = config.get("http_max_connections", 512)
        self.http_max_keepalive = config.get("http_max_keepalive", 256)

        # Lazily-created per-event-loop httpx client + semaphore. We can't
        # build them in __init__ because no event loop is running at that
        # point, and httpx.AsyncClient binds to the current loop on first use.
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._async_init_lock = threading.Lock()

        # Ray-based execution pool (legacy path, only built when async is off)
        self.execution_pool = None
        if not self.use_async_http:
            self.execution_pool = init_search_execution_pool(
                num_workers=self.num_workers,
                enable_global_rate_limit=self.enable_global_rate_limit,
                rate_limit=self.rate_limit,
                mode=PoolMode.ThreadMode,
            )

        # Retrieval service configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = config.get("topk", 3)
        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")

        logger.info(f"Initialized SearchTool with config: {config}")

    def _ensure_async_resources(self):
        """Lazily create the httpx client + semaphore on first use."""
        if self._async_client is not None and self._async_semaphore is not None:
            return
        with self._async_init_lock:
            if self._async_client is None:
                limits = httpx.Limits(
                    max_connections=self.http_max_connections,
                    max_keepalive_connections=self.http_max_keepalive,
                )
                self._async_client = httpx.AsyncClient(
                    limits=limits,
                    timeout=httpx.Timeout(self.timeout),
                    http2=False,
                )
            if self._async_semaphore is None:
                self._async_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        """Execute search operation using retrieval service.

        Args:
            instance_id: Tool instance ID
            query_list: List of search queries
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Execute search either via the in-process async HTTP client (default,
        # fast) or via the legacy Ray actor pool (kept for backwards compat).
        try:
            if self.use_async_http:
                self._ensure_async_resources()
                result_text, metadata = await async_perform_single_search_batch(
                    client=self._async_client,
                    retrieval_service_url=self.retrieval_service_url,
                    query_list=query_list_from_params,
                    topk=self.topk,
                    semaphore=self._async_semaphore,
                    timeout=timeout,
                )
            else:
                result_text, metadata = await self.execution_pool.execute.remote(
                    self.execute_search,
                    instance_id,
                    query_list_from_params,
                    self.retrieval_service_url,
                    self.topk,
                    timeout,
                )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
