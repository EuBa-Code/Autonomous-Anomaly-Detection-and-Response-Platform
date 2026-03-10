from fastmcp import FastMCP
from qdrant.src import compression_retriever, format_docs

mcp = FastMCP('MCP_server')

@mcp.tool()
def retrive_context(query: str):
    retrieved_docs = compression_retriever.invoke(query)
    formatted_docs = format_docs(retrieved_docs)

    return formatted_docs

if __name__ == '__main__':
    mcp.run(transport='http', host='0.0.0.0', port=9000)