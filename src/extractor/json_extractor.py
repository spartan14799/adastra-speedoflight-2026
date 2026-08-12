import abc
import json
import re
import os
import uuid
from typing import Any, Dict, List
from langdetect import detect, LangDetectException

from .base import BaseExtractor


class JSONExtractor(BaseExtractor):
    """
    Concrete class for JSON file ingestion.
    Includes logic for omitting heavy metadata (images) and 
    now explicitly injects links into the raw text.
    """

    @property
    def tipo_fuente(self) -> str:
        return "json"

    def _process_article(self, item: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = str(item.get("doc_id") or item.get("id") or uuid.uuid4())
        
        # Priority extraction of the title for the top level
        title = str(item.get("title") or item.get("titulo") or "Untitled").strip()
        
        text_blocks = []
        if title != "Untitled":
            text_blocks.append(f"{title}.")
            
        if "sections" in item and isinstance(item["sections"], list):
            for sec in item["sections"]:
                if isinstance(sec, dict):
                    if "heading" in sec and sec["heading"]:
                        text_blocks.append(f"{str(sec['heading']).strip()}.")
                    if "paragraphs" in sec and isinstance(sec["paragraphs"], list):
                        paragraphs = [str(p).strip() for p in sec["paragraphs"] if str(p).strip()]
                        text_blocks.append(" ".join(paragraphs))
                        
        if "lists" in item and isinstance(item["lists"], list):
            list_items = [f"{str(li).strip()}." for li in item["lists"] if str(li).strip()]
            text_blocks.append(" ".join(list_items))

        if item.get("body_text"):
            text_blocks.append(str(item["body_text"]).strip())
            
        if item.get("body_paragraphs"):
            paragraphs = item["body_paragraphs"]
            if isinstance(paragraphs, list):
                text_blocks.append(" ".join(str(p).strip() for p in paragraphs if str(p).strip()))
            else:
                text_blocks.append(str(paragraphs).strip())

        # =========================================================================
        # Dynamic metadata configuration and link injection
        # =========================================================================
        text_keys = {"title", "titulo", "sections", "lists", "body_text", "body_paragraphs", "id", "doc_id"}
        image_patterns = {"image", "img", "thumbnail", "picture", "figure", "photo", "url", "cover", "portada"}
        metadata = {}
        
        for k, v in item.items():
            k_lower = str(k).lower()
            
            # Filter to exclude keys indicating visual content (images)
            is_image = any(pattern in k_lower for pattern in image_patterns) if "link" not in k_lower else False
            
            # Scenario 1: List of link dictionaries -> Goes straight to text_blocks
            if isinstance(v, list) and len(v) > 0 and all(isinstance(i, dict) and "url" in i for i in v):
                for element in v:
                    link_text = str(element.get("text", "Link")).strip()
                    link_url = str(element.get("url", "")).strip()
                    if link_url:
                        text_blocks.append(f"{link_text}: {link_url}.")

            # Scenario 2: Single link dictionary -> Goes straight to text_blocks
            elif isinstance(v, dict) and "url" in v:
                link_text = str(v.get("text", "Link")).strip()
                link_url = str(v.get("url", "")).strip()
                if link_url:
                    text_blocks.append(f"{link_text}: {link_url}.")

            # Original behavior for the rest of legitimate metadata
            elif k not in text_keys and not is_image:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    metadata[k] = v
                else:
                    # Strict serialization for nested arrays
                    metadata[k] = json.dumps(v, ensure_ascii=False)
                    
        # Assemble raw text AT THE END to include the extracted links
        raw_text = " ".join(text_blocks)
        metadata["format"] = "json"
        
        return {
            "doc_id": doc_id,
            "title": title,
            "raw_text": raw_text,
            "metadata": metadata
        }
    
    def _unpack_objects(self, data: Any) -> List[Dict[str, Any]]:
        documents = []
        if isinstance(data, dict):
            documents.append(self._process_article(data))
        elif isinstance(data, list):
            for element in data:
                documents.extend(self._unpack_objects(element))
        else:
            documents.append({
                "doc_id": str(uuid.uuid4()),
                "title": "Untitled",
                "raw_text": str(data),
                "metadata": {"format": "json"}
            })
        return documents

    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        raw_documents = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)
                try:
                    json.loads(first_line)
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            raw_documents.extend(self._unpack_objects(obj))
                except json.JSONDecodeError:
                    f.seek(0)
                    complete_data = json.load(f)
                    raw_documents.extend(self._unpack_objects(complete_data))
        except Exception as e:
            raise RuntimeError(f"Processing failure in {file_path}: {str(e)}")
        return raw_documents


