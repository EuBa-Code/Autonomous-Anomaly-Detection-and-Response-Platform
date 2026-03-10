import json
import asyncio  
from fastapi import FastAPI
from fastapi.responses import StreamingResponse 
from langchain_core.messages import HumanMessage  
from langchain_service.src import build_agent, ChatRequest


app = FastAPI()

_agent = None
_agent_lock = asyncio.Lock()


async def get_agent():
    global _agent
    if _agent is None:
        async with _agent_lock:
            if _agent is None:
                _agent = await build_agent()
    return _agent

# Server-Sent Events
def sse_pack(event: str, data) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'

@app.post('/chat/stream')
async def chat_stream(req: ChatRequest):
    agent = await get_agent()

    async def event_generator():
        yield sse_pack('status', {'state':'started'})

        async for event in agent.astream_events(
            {'messages':
             [HumanMessage(content=req.message)]
             },
             version='v2'
        ):
            e_type = event.get('event')

            if e_type == 'on_chat_model_stream':
                chunk = event['data'].get('chunk')
                if chunk and getattr(chunk, 'context', None):
                    yield sse_pack('token', {'text': chunk.context})
