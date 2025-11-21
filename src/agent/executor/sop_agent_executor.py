from __future__ import annotations

import json
import logging
import sqlite3
import mlflow
from pathlib import Path
from typing import Dict, Any, List, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, BaseMessage

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from src.agent.integrations.databricks_vector_search import DatabricksVectorRetriever
from src.agent.utils.llm_provider import get_llm_model
from src.agent.config.settings import settings

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    search_query: str
    retrieved_context: str
    final_response: str
    tenant_id: str

class SOPAgentExecutor(AgentExecutor):
    def __init__(
        self, config_path: str | None = None, config_dict: Dict[str, Any] | None = None
    ) -> None:
        """
        Initialize SOP Agent Executor with tenant-aware configuration.
        """
        # 1. Load multi-tenant configuration
        if config_dict is not None:
            self.tenant_configs = config_dict
        elif config_path is not None:
            self.tenant_configs = self._load_config(config_path)
        else:
            default_path = str(
                Path(__file__).parent.parent.parent.parent
                / "config"
                / "sop_config.json"
            )
            self.tenant_configs = self._load_config(default_path)

        # 2. Setup MLflow Tracing (Replaces Langfuse)
        # This automatically traces LangChain execution to Databricks Inference Tables
        mlflow.langchain.autolog()

        # 3. Setup Persistence (Unity Catalog Volume)
        # We use SQLite stored on a Volume for persistence across restarts
        try:
            db_path = settings.checkpoint_path
            # Ensure parent directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # check_same_thread=False is required for Uvicorn async environments
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.checkpointer = SqliteSaver(self.conn)
            logger.info(f"Persistence initialized at {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize persistence at {settings.checkpoint_path}: {e}")
            logger.warning("Falling back to MemorySaver (History will be lost on restart)")
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        """Get configuration for a specific tenant."""
        logger.info(f"Getting configuration for tenant: {tenant_id}")
        if "tenants" in self.tenant_configs:
            if tenant_id not in self.tenant_configs["tenants"]:
                # Fallback to default if tenant not found, or raise error based on preference
                logger.warning(f"Tenant {tenant_id} not found, falling back to 'default'")
                return self.tenant_configs["tenants"]["default"]
            return self.tenant_configs["tenants"][tenant_id]
        else:
            return self.tenant_configs

    def _build_graph(
        self, tenant_config: Dict[str, Any], retrieval_llm: Any, response_llm: Any
    ) -> StateGraph:
        """Build the agent graph with tenant-specific configuration."""

        def generate_search_query(state: AgentState) -> AgentState:
            messages = state["messages"]
            user_query = messages[-1].content

            # Context window management (last 5 messages)
            conversation_text = "\n".join(
                [
                    f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                    for m in messages[-6:-1]
                ]
            )

            retrieval_prompt = tenant_config.get("retrieval_prompt", "Query: {search_query}")
            prompt_text = retrieval_prompt.replace("{conversation}", conversation_text)
            prompt_text = prompt_text.replace("{search_query}", user_query)

            response = retrieval_llm.invoke(prompt_text)
            search_query = str(response.content).strip()
            
            logger.info(f"Generated search query: {search_query}")
            return {**state, "search_query": search_query}

        def retrieve_context(state: AgentState) -> AgentState:
            search_query = state["search_query"]
            
            # Extract Vector Search Config from Tenant JSON
            vs_config = tenant_config.get("vector_search", {})
            
            # Initialize Retriever with Tenant-specific Index/Endpoint
            retriever = DatabricksVectorRetriever(
                endpoint_name=vs_config.get("endpoint_name"),
                index_name=vs_config.get("index_name"),
                column_map=vs_config.get("columns")
            )
            
            results = retriever.query(
                search_query,
                top_k=tenant_config.get("retrieval_config", {}).get("top_k", 7)
            )

            context_lines = []
            if results:
                for idx, r in enumerate(results, 1):
                    source = r.metadata.get("source", "Unknown")
                    context_lines.append(
                        f"Document {idx}:\nSource: {source}\nContent: {r.text}\nRelevance: {r.score:.3f}"
                    )

            retrieved_context = "\n\n".join(context_lines) if context_lines else "No relevant policy information found."
            
            logger.info(f"Retrieved {len(context_lines)} documents")
            return {**state, "retrieved_context": retrieved_context}

        def generate_response(state: AgentState) -> AgentState:
            messages = state["messages"]
            user_query = messages[-1].content
            retrieved_context = state["retrieved_context"]

            conversation_text = "\n".join(
                [
                    f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                    for m in messages[-6:]
                ]
            )

            rag_prompt_template = tenant_config.get("rag_prompt", "{context}")
            prompt_text = rag_prompt_template.replace("{conversation}", conversation_text)
            prompt_text = prompt_text.replace("{context}", retrieved_context)
            prompt_text = prompt_text.replace("{search_query}", user_query)

            logger.info(f"Generating response using model")
            response = response_llm.invoke(prompt_text)
            final_response = str(response.content).strip()

            return {**state, "final_response": final_response}

        workflow = StateGraph(AgentState)

        workflow.add_node("generate_search_query", generate_search_query)
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_response", generate_response)

        workflow.set_entry_point("generate_search_query")
        workflow.add_edge("generate_search_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)

        # Compile with the SqliteSaver checkpointer
        return workflow.compile(checkpointer=self.checkpointer)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute the agent.
        """
        user_input = context.get_user_input()

        if not user_input or not isinstance(user_input, str):
            await event_queue.enqueue_event(new_agent_text_message("Please provide a text question."))
            return

        tenant_id = self._extract_tenant_id(context)
        
        try:
            # Load Tenant Config
            tenant_config = self._get_tenant_config(tenant_id)
            llm_config = tenant_config.get("llm_config", {})
            
            # Initialize Databricks Models (ChatDatabricks)
            retrieval_llm = get_llm_model(
                "databricks", 
                llm_config.get("retrieval_model"), 
                temperature=llm_config.get("retrieval_temperature", 0.1)
            )
            response_llm = get_llm_model(
                "databricks", 
                llm_config.get("response_model"), 
                temperature=llm_config.get("response_temperature", 0.7)
            )

            # Build and Run Graph
            graph = self._build_graph(tenant_config, retrieval_llm, response_llm)
            
            thread_id = self._extract_thread_id(context)
            config = {"configurable": {"thread_id": f"{tenant_id}_{thread_id}"}} if thread_id else {}

            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_input)],
                "search_query": "",
                "retrieved_context": "",
                "final_response": "",
                "tenant_id": tenant_id,
            }

            final_state = graph.invoke(initial_state, config=config)
            response = final_state.get("final_response", "I could not generate a response.")

            await event_queue.enqueue_event(new_agent_text_message(response))

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_agent_text_message("I apologize, but an error occurred while processing your request.")
            )

    def _extract_tenant_id(self, context: RequestContext) -> str:
        """Extract tenant ID from request context metadata or headers."""
        try:
            # 1. Context Metadata (A2A standard)
            metadata = getattr(context, "metadata", None)
            if metadata and isinstance(metadata, dict) and "tenant_id" in metadata:
                return str(metadata["tenant_id"])

            # 2. Request Headers
            request_data = getattr(context, "request_data", {})
            if isinstance(request_data, dict):
                if "tenant_id" in request_data:
                    return str(request_data["tenant_id"])
                
                headers = request_data.get("headers", {})
                if isinstance(headers, dict):
                    tid = headers.get("X-Tenant-ID") or headers.get("x-tenant-id")
                    if tid: return str(tid)

            return "default"
        except Exception:
            return "default"

    def _extract_thread_id(self, context: RequestContext) -> str:
        """Extract thread ID for conversation persistence."""
        try:
            metadata = getattr(context, "metadata", None)
            if metadata and isinstance(metadata, dict) and "thread_id" in metadata:
                return str(metadata["thread_id"])
            
            request_data = getattr(context, "request_data", {})
            if isinstance(request_data, dict) and "thread_id" in request_data:
                return str(request_data["thread_id"])

            return "default_thread"
        except Exception:
            return "default_thread"

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("Task cancellation not supported")