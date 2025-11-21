import logging
# [CHANGED] Import from community to avoid mlflow conflict
from langchain_community.chat_models import ChatDatabricks

logger = logging.getLogger(__name__)

def get_llm_model(model_provider: str, model_name: str, temperature: float = 0.7):
    logger.info(f"Initializing Databricks model endpoint: {model_name}")
    
    return ChatDatabricks(
        endpoint=model_name, 
        temperature=temperature
    )