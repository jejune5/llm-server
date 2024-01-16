# coding=utf-8
# Implements API for chatcare in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python api.py
# code ref: https://github.com/QwenLM/Qwen-7B/blob/main/openai_api.py
# Visit http://localhost:8000/docs (default) for documents.

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from server.types import *
from config import params
from chat_api import chat_completions, chat_direct_with_llm


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def create_app():
    app = FastAPI(
        title="LLM API Server",
        lifespan=lifespan,
    )
    # 跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Tag: Chat

    app.post(
        "/v1/chat/completions",
        tags=["Chat"],
        summary="LLM",
        response_model=ChatCompletionResponse
    )(chat_completions)

    app.post(
        "/v1/chat/llm",
        tags=["Chat"],
        summary="(DEV) 直接与 LLM 对话",
        response_model=ChatCompletionResponse
    )(chat_direct_with_llm)

    return app


app = create_app()


def run_api():
    uvicorn.run(
        app,
        host=params.api_host,
        port=params.api_port,
        workers=1
    )


if __name__ == '__main__':
    run_api()
