# coding=utf-8
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from server.types import *
from tools.logger import logger
from config import params
from chat_llm import load_llm_model, chat_llm, chat_llm_stream




async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """LLM对话接口"""
    if params.chat_mode == "vs":
        chat_completions_api = chat_direct_with_search_engine
    elif params.chat_mode == "llm":
        chat_completions_api = chat_direct_with_llm
    try:
        return await chat_completions_api(request)
    except:
        logger.exception('An error occurred in api: `/v1/chat/completions`!')
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="很抱歉，请重新问我一次！"),
            finish_reason="stop"
        )
        return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def chat_stream_search_engine_generator(query: str, history: List[List[str]] = None):
    async for chunk in chat_vector_search_stream(query, history):
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=chunk),
            finish_reason="stop"
        )
        chunk = ChatCompletionResponse(model="se_stream", choices=[choice_data], object="chat.completion")
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def chat_direct_with_search_engine(request: ChatCompletionRequest):
    """
    直接与向量数据库对话，暂不支持历史、多轮对话
    """
    # 用户名默认为user
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content
    if request.stream:
        return StreamingResponse(chat_stream_search_engine_generator(query), media_type="text/event-stream")
    content = await chat_vector_search(query)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=content),
        finish_reason="stop"
    )
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def chat_stream_llm_generator(query: str, history: List[List[str]] = None):
    async for chunk in chat_llm_stream(query, history):
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=chunk),
            finish_reason="stop"
        )
        chunk = ChatCompletionResponse(model="llm_stream", choices=[choice_data], object="chat.completion")
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def chat_direct_with_llm(request: ChatCompletionRequest):
    """直接与LLM对话，暂不支持历史、多轮对话"""
    global model, tokenizer, infer, infer_stream
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content
    prev_messages = request.messages[:-1]
    # Temporarily, the system role does not work as expected. We advise that you write the setups for role-play in your query.
    # if len(prev_messages) > 0 and prev_messages[0].role == "system":
    #     query = prev_messages.pop(0).content + query
    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i + 1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i + 1].content])
            else:
                raise HTTPException(status_code=400, detail="Invalid request.")
    else:
        raise HTTPException(status_code=400, detail="Invalid request.")
    if 'infer' not in globals():
        model, tokenizer, infer, infer_stream = load_llm_model(params)
    if request.stream:
        return StreamingResponse(chat_stream_llm_generator(query, history), media_type="text/event-stream")
    content = await chat_llm(query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=content),
        finish_reason="stop"
    )
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")
