from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from config import inference_settings

client = MultiServerMCPClient(
    {
        'mcp_client': {
            'transport': 'http',
            'url': inference_settings.mcp_server_uri,
            'headers': { 
                'X-Custom-Header':'custom-value'
            }
        }
    }
)