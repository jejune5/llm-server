from transformers import AutoTokenizer, AutoModel
from tools.logger import logger


def load_model(checkpoint_path, device="cuda"):
    if device != "cuda":
        logger.error(f"ChatGLM2-6b-int4 is not allowed on {device=}!")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True).half().to(device)
    model = model.eval()
    return model, tokenizer


def infer(model, tokenizer, query: str, history: list = None, **kwargs):
    if history:
        raise NotImplementedError
    content, history = model.chat(tokenizer, query, history=history, **kwargs)
    return content


def infer_stream(model, tokenizer, query: str, history: list = None, top_p: float = 1.0, temperature: float = 1.0,
                 **kwargs):
    """
    流式推理
    :return:
    """
    for content, query_content in model.stream_chat(tokenizer, query, history, top_p=top_p,
                                                    temperature=temperature, **kwargs):
        yield content


if __name__ == '__main__':
    checkpoint_path = r'./models/chatglm2-6b-int4'
    model, tokenizer = load_model(checkpoint_path, device='cuda')
    print('direct', '-' * 88)
    content = infer(model, tokenizer, "你好", top_p=2, temperature=1)
    print(content)

