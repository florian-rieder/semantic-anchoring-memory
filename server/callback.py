from typing import Any

from langchain.callbacks.base import AsyncCallbackHandler

from server.schemas import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses token by token."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.dict())
    
    async def on_chat_model_start(*args: Any, **kwards: Any):
        # Implementation of on_chat_model_start is necessary !
        # But we don't need it.
        pass
