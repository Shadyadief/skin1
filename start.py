"""
Startup script — initializes RAG agent then launches FastAPI server.
"""
import asyncio
import os
import sys
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("dermai.start")


async def initialize():
    """Pre-warm agents before serving requests."""
    logger.info("Initializing DermAI Pro Multi-Agent System...")

    from agents.rag_agent import rag_agent
    await rag_agent.initialize()

    logger.info("All agents ready. Starting server.")


def main():
    port = int(os.environ.get("PORT", 8090))

    async def run():
        await initialize()
        config = uvicorn.Config(
            "main:app",
            host="0.0.0.0",
            port=port,
            log_level="info",
            reload=False,
        )
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(run())


if __name__ == "__main__":
    main()
