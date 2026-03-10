import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from src import notify_operator, ChatRequest, build_agent

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


def sse_pack(event: str, data) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/chat/stream')
async def chat_stream(req: ChatRequest):
    agent = await get_agent()

    async def event_generator():
        yield sse_pack('status', {'state': 'started'})
        
        # Initialize an accumulator for the final summary
        full_text_list = []

        async for event in agent.astream_events(
            {'messages': [HumanMessage(content=req.message)]},
            version='v2'
        ):
            e_type = event['event']

            # Handle Model Tokens
            if e_type == 'on_chat_model_stream':
                chunk = event['data'].get('chunk')
                if chunk and chunk.content:
                    content = chunk.content
                    full_text_list.append(content)
                    yield sse_pack('token', {'text': content})

            # Handle Tooling
            elif e_type == 'on_tool_start':
                yield sse_pack('tool_start', {'name': event['name']})
            
            elif e_type == 'on_tool_end':
                # Optional: capture tool output for the log/summary
                tool_output = event['data'].get('output')
                yield sse_pack('tool_end', {'name': event['name'], 'output': str(tool_output)})

        # Join the fragments for the final summary
        full_response = "".join(full_text_list)
        
        yield sse_pack('done', {'ok': True})

        # Notify operator without blocking the final 'done' message
        # Ensure notify_operator is either async or wrapped in a thread
        try:
            notify_operator(machine_id=req.machine_id, summary=full_response)
        except Exception as e:
            print(f"Failed to notify operator: {e}")

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no', # Crucial for Nginx proxying
        },
    )