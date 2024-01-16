# coding=utf-8

from server.types import *
from tools.logger import logger
from config import params


def load_llm_model(params):
    """ 载入不同llm模型用 """
    global model, tokenizer, infer, infer_stream
    llm_model_name, llm_checkpoint_dir = params.llm_model_name, params.llm_checkpoint_dir
    model, tokenizer, infer = None, None, None
    if llm_model_name == "baichuan13b":
        from repo.llm_api_server.llms.qwen import load_model, infer, infer_stream
        model = load_model(params.llm_checkpoint_dir, params.device)
    else:
        from repo.llm_api_server.llms.chatglm2_6B_int4 import load_model, infer, infer_stream
        model, tokenizer = load_model(params.llm_checkpoint_dir, params.device)

    if params.debug:
        logger.info(
            f"Load llm model successfully! || llm_model_name: {llm_model_name} || llm_checkpoint_dir: {llm_checkpoint_dir} || type: {type(model)}")
    return model, tokenizer, infer, infer_stream


# 启动预加载模型
if params.chat_mode == "llm":
    model, tokenizer, infer, infer_stream = load_llm_model(params)


async def chat_llm(query: str, history: Optional[List[List[str]]] = None) -> str:
    """
    llm单次推理，非流式
    :param query:
    :param history:
    :return:
    """
    time_start = time.time()
    global model, tokenizer, infer
    if params.llm_model_name == "baichuan13b":
        content = infer(model, query, history)
    else:
        content = infer(model, tokenizer, query, history)
    content = content.strip()
    if params.debug:
        logger.info(
            f"Chat with llm successfully! || Cost_time(s): {time.time() - time_start} || Query: {query} || Content: {content}")
    return content


async def chat_llm_stream(query: str, history: Optional[List[List[str]]] = None):
    """
    llm流式推理
    :param query:
    :param history:
    :return:
    """
    content = ""
    time_start = time.time()
    global model, tokenizer, infer
    if params.llm_model_name == "baichuan13b":
        content_generator = infer_stream(model, query, history)
    else:
        content_generator = infer_stream(model, tokenizer, query, history)

    for chunk in content_generator:
        chunk = chunk.replace(content, "")
        content += chunk
        yield chunk

    if params.debug:
        logger.info(
            f"Chat stream with llm successfully! || Cost_time(s): {time.time() - time_start} || Query: {query} || Content: {content}")
