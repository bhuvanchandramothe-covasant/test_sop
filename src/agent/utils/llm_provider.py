import logging
from langchain_databricks import ChatDatabricks

logger = logging.getLogger(__name__)

def get_llm_model(model_provider: str, model_name: str, temperature: float = 0.7):
    # 'model_provider' is kept for signature compatibility but ignored
    logger.info(f"Initializing Databricks model endpoint: {model_name}")
    
    return ChatDatabricks(
        endpoint=model_name, 
        temperature=temperature
    )