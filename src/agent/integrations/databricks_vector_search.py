from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from databricks.vector_search.client import VectorSearchClient
from src.agent.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    id: str
    score: float
    text: Optional[str]
    metadata: Dict[str, Any]

class DatabricksVectorRetriever:
    def __init__(self, endpoint_name: str = None, index_name: str = None, column_map: Dict[str, str] = None):
        # Use passed values, or fallback to settings
        self.endpoint_name = endpoint_name or settings.default_vs_endpoint_name
        self.index_name = index_name or settings.default_vs_index_name
        
        # Default column mapping updated to match your schema
        self.column_map = column_map or {
            "id": "id", 
            "text": "chunk_text", 
            "source": "source_path"
        }

        self.client = VectorSearchClient(disable_notice=True)
        self.index = self.client.get_index(
            endpoint_name=self.endpoint_name,
            index_name=self.index_name
        )

    def query(self, query_text: str, top_k: int = 7, score_threshold: float = 0.0) -> List[VectorSearchResult]:
        logger.info(f"Querying {self.index_name} on {self.endpoint_name}")
        
        try:
            # We request the columns defined in our mapping
            req_cols = list(self.column_map.values())
            
            results = self.index.similarity_search(
                query_text=query_text,
                columns=req_cols,
                num_results=top_k
            )
            
            if not results or 'result' not in results:
                return []

            data = results['result']['data_array']
            manifest = results['manifest']['columns']
            col_idx_map = {col['name']: i for i, col in enumerate(manifest)}
            
            parsed_results = []
            for row in data:
                score = row[-1]
                if score < score_threshold:
                    continue

                # Dynamic extraction based on config mapping
                text_col = self.column_map["text"]
                source_col = self.column_map["source"]
                id_col = self.column_map["id"]

                text_val = row[col_idx_map.get(text_col)] if text_col in col_idx_map else ""
                source_val = row[col_idx_map.get(source_col)] if source_col in col_idx_map else ""
                id_val = row[col_idx_map.get(id_col)] if id_col in col_idx_map else ""

                parsed_results.append(VectorSearchResult(
                    id=str(id_val),
                    score=score,
                    text=str(text_val),
                    metadata={"source": source_val}
                ))
                
            return parsed_results

        except Exception as e:
            logger.error(f"Vector Search failed: {e}")
            return []