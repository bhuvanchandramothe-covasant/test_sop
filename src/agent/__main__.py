import asyncio
import logging
import sys
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, AgentProvider
from starlette.responses import JSONResponse
from .config.settings import settings
from .executor.sop_agent_executor import SOPAgentExecutor
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Fix for Windows: Use SelectorEventLoop for psycopg compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def create_agent_card() -> AgentCard:
    skills = [
        AgentSkill(
            id="policy_inquiry",
            name="Store Policy Information",
            description="""Get information about store policies, procedures, and
            guidelines for overall operations or specific product categories""",
            tags=["policy", "procedures", "guidelines", "sop"],
            examples=[
                "What is the return policy for electronics?",
                "How do I handle customer complaints?",
                "What are the store opening procedures?",
            ],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        ),
        AgentSkill(
            id="sop_search",
            name="Standard Operating Procedure Search",
            description="""Search for specific standard operating procedures and
            operational guidelines""",
            tags=["sop", "operations", "procedures"],
            examples=[
                "What is the SOP for handling returns?",
                "Show me the procedure for inventory management",
                "What are the safety protocols?",
            ],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        ),
        AgentSkill(
            id="category_policy",
            name="Category-Specific Policies",
            description="Get policy information specific to product categories",
            tags=["policy", "category", "products"],
            examples=[
                "What are the policies for sporting goods?",
                "Tell me about alcohol sales policies",
                "What are the guidelines for handling perishable items?",
            ],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        ),
    ]

    return AgentCard(
        name=settings.agent_name,
        description=f"""{settings.agent_description} | RAG-powered policy search
        using GCP Vertex AI RAG Corpus and OpenAI | Supports multi-session
        conversations with PostgreSQL persistence""",
        url=settings.agent_url,
        version=settings.agent_version,
        protocol_version="0.3.0",
        preferred_transport="HTTP+JSON",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=settings.agent_streaming,
            push_notifications=settings.agent_push_notifications,
        ),
        skills=skills,
        supports_authenticated_extended_card=False,
        provider=AgentProvider(
            organization="Store Operations", url="https://example.com"
        ),
        documentation_url=None,
        icon_url=None,
    )


def main():
    setup_logging()

    agent_card = create_agent_card()

    logger.info(f"Starting {settings.agent_name}")
    logger.info(f"Description: {settings.agent_description}")
    logger.info(f"Agent URL: {settings.agent_url}")
    logger.info(f"Agent Card: {settings.agent_url}/.well-known/agent-card.json")
    logger.info("A2A Protocol (JSON-RPC):")
    logger.info(f"  Endpoint: POST {settings.agent_url}/")
    logger.info("  Method: message/send")
    logger.info("Test the agent: python test_a2a_client.py")
    logger.info(
        "Tech Stack: GCP Vertex AI RAG Corpus (retrieval) + "
        "OpenAI (LLM) + PostgreSQL (persistence)"
    )
    logger.debug(f"Agent host: {settings.agent_host}, port: {settings.agent_port}")
    logger.debug(f"Log level: {settings.log_level}")

    request_handler = DefaultRequestHandler(
    agent_executor=SOPAgentExecutor(),
    task_store=InMemoryTaskStore(),
    )

    server_app = A2AStarletteApplication(
        agent_card=create_agent_card(),
        http_handler=request_handler,
    )
    # Build the app and add health endpoint
    app = server_app.build()

    @app.route("/")
    async def health(request):
        return JSONResponse({"status": "ok"})

    uvicorn.run(
        app,
        host=settings.agent_host,
        port=settings.agent_port,
        # log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
