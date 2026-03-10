from fastmcp import FastMCP
from qdrant.src import compression_retriever, format_docs
from mongodb.src import MongoLogger

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP('MCP_server')
mongo_logger = MongoLogger()

@mcp.tool()
def retrive_context(query: str, machine_id: int):

    try:
        retrieved_docs = compression_retriever.invoke(query)
        formatted_docs = format_docs(retrieved_docs)
    except Exception as e:
        logging.error(f"Error while retrieving documents with Qdrant, error: {e}")        
        return f'Error retrieving context with Qdrant: {e}'

    try:
        mongo_logger.log_query(
            machine_id = machine_id
        )
    except Exception as e:
        logging.error(f"Error while saving logs with MongoDB, error: {e}")        
        return f'Error saving logs with MongoDB: {e}'

    return formatted_docs

if __name__ == '__main__':
    mcp.run(transport='http', host='0.0.0.0', port=8020)