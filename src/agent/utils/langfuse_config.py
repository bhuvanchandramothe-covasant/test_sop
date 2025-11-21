import dotenv
from langfuse import get_client

dotenv.load_dotenv()


def init_langfuse():
    return get_client()
