import json
import logging
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from websockets.exceptions import ConnectionClosedOK
from langchain.chains import ConversationChain

from chat import get_chain
from server.callback import StreamingLLMCallbackHandler
from server.schemas import ChatResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = FastAPI()

app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        context={'request': request}, name="index.html"
    )


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)

    stream_handler = StreamingLLMCallbackHandler(websocket)
    conversation = get_chain(stream_handler)

    def end_conversation(conversation: ConversationChain):
        """
        At the end of the conversation, pass the conversation history through
        the memory creation pipeline.
        """
        # Get the conversation history from the conversation memory
        chat_history = conversation.memory.memories[0].chat_memory
        # Memorize the conversation
        conversation.memory.memories[1].memorize(str(chat_history))

    while True:
        # Mostly lifted out of https://github.com/pors/langchain-chat-websockets
        try:
            message = await websocket.receive_text()
            data = json.loads(message)
            message_type = data.get("type")

            # Handle setting the memory model
            if message_type == "memory_model":
                memory_model = data.get("value", None)
                # Create new chain that uses the chosen memory model
                conversation = get_chain(stream_handler, memory_model)
                continue

            # Handle when the user pressed the "end conversation" button
            if message_type == "end_conversation":
                logger.info('User ended the conversation')
                end_conversation(conversation)
                # reset conversation
                conversation = get_chain(stream_handler)
                continue

            # Handle unknown message types
            if message_type != 'message':
                raise ValueError('Unknown type received: {message_type}')

            # Receive and send back the client message
            user_msg = data.get('value')

            resp = ChatResponse(
                sender="human", message=user_msg, type="stream")
            # await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            # Send the message to the chain and feed the response back to the client
            # the stream handler will send chunks as they come
            await conversation.acall({"input": user_msg})

            # Send the end-response back to the client
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

        except WebSocketDisconnect:
            logging.info("WebSocketDisconnect")
            # TODO try to reconnect with back-off
            manager.disconnect(websocket)
            # end_conversation(conversation)
            break

        except ConnectionClosedOK:
            logging.info("ConnectionClosedOK")
            # TODO handle this?
            # end_conversation(conversation)
            break

        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}
