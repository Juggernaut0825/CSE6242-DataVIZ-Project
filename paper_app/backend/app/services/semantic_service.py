from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from app.config import settings
from app.openrouter_client import OpenRouterClient


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "this", "that", "these",
    "those", "with", "from", "into", "about", "for", "have", "has", "had", "will",
    "would", "could", "should", "they", "them", "their", "there", "here", "what",
    "when", "where", "which", "who", "how", "why", "you", "your", "our", "ours",
    "his", "her", "hers", "its", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "as", "it", "we", "i",
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SemanticService:
    def __init__(self) -> None:
        self.client = OpenRouterClient() if settings.llm_api_key else None

    async def extract_bundle(self, text: str, allow_llm: bool = True) -> Dict[str, Any]:
        llm_bundle = await self._extract_bundle_with_llm(text) if allow_llm else None
        if llm_bundle:
            return llm_bundle
        return self._extract_bundle_fallback(text)

    async def _extract_bundle_with_llm(self, text: str) -> Dict[str, Any] | None:
        if not self.client or not text.strip():
            return None
        try:
            response = await asyncio.wait_for(
                self.client.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You extract graph-ready memory labels from text for a mind-map style graph. "
                                "Return one JSON object only. "
                                "Every display_label must look like a compact concept-map title, not a sentence fragment. "
                                "For claims, produce a short proposition headline in 3-6 words. "
                                "Avoid leading filler words such as first, this, it, we, there, here. "
                                "Avoid copying long spans from the source text. "
                                "Prefer noun phrases or concise proposition titles. "
                                "For entities and concepts, keep labels very short, usually 1-3 words. "
                                "Use only these relation types: SUPPORTS, CONTRADICTS, CAUSES, "
                                "TEMPORAL_BEFORE, ELABORATES, ASSOCIATED_WITH. "
                                "The source_claim and target_claim in relations must exactly match claim text values."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Extract a compact semantic bundle from the following text.\n"
                                "Return JSON with this schema:\n"
                                "{\n"
                                '  "claims": [{"text": string, "display_label": string, "entities": [string], "concepts": [string]}],\n'
                                '  "entities": [{"text": string, "display_label": string}],\n'
                                '  "concepts": [{"text": string, "display_label": string}],\n'
                                '  "relations": [{"source_claim": string, "target_claim": string, "type": string}]\n'
                                "}\n"
                                "Constraints:\n"
                                "- At most 3 claims, 10 entities, 10 concepts, 6 relations.\n"
                                "- Claim text can be a lightly normalized statement from the text.\n"
                                "- Every claim display_label must be 3-6 words.\n"
                                "- Every display_label must read like a mind-map node title.\n"
                                "- Do not start display_label with discourse markers such as first, second, this, it, we.\n"
                                "- display_label must be much shorter than the full text, not just slightly shorter.\n"
                                "- Do not include commentary or markdown.\n\n"
                                f"Text:\n{text[:5000]}"
                            ),
                        },
                    ],
                    model=settings.llm_model,
                    temperature=0.1,
                    max_tokens=1400,
                    stream=False,
                    enable_reasoning=False,
                    response_format={"type": "json_object"},
                ),
                timeout=12.0,
            )
            content = self._response_text(response)
            if not content:
                return None
            parsed = json.loads(content)
            return self._normalize_llm_bundle(parsed, text)
        except asyncio.TimeoutError:
            logger.info("semantic llm extraction timed out; using fallback")
            return None
        except Exception:
            return None

    def _extract_bundle_fallback(self, text: str) -> Dict[str, Any]:
        sentences = self._split_sentences(text)
        claims = [sentence for sentence in sentences[:3] if sentence]
        claim_details = [self._labels_for_span(claim) for claim in claims]
        entities = self._extract_entities(text)
        concepts = self._extract_concepts(text)
        relations = self._extract_internal_relations(sentences)
        return {
            "claims": claims,
            "claim_details": claim_details,
            "entities": entities,
            "concepts": concepts,
            "relations": relations,
            "display_labels": {
                "claims": {claim: self._make_display_label("claim", claim) for claim in claims},
                "entities": {entity: self._make_display_label("entity", entity) for entity in entities},
                "concepts": {concept: self._make_display_label("concept", concept) for concept in concepts},
            },
        }

    def classify_unit_relation(
        self,
        left_text: str,
        left_labels: Dict[str, Any],
        right_text: str,
        right_labels: Dict[str, Any],
    ) -> str | None:
        left_tokens = set(left_labels.get("concepts", [])) | set(left_labels.get("entities", []))
        right_tokens = set(right_labels.get("concepts", [])) | set(right_labels.get("entities", []))
        overlap = left_tokens & right_tokens
        joined = f"{left_text.lower()} {right_text.lower()}"
        has_explicit_signal = self._contains_any(
            joined,
            (
                "because", "therefore", "thus", "result", "leads to", "causes", "due to",
                "before", "after", "then", "next", "earlier", "later",
                "however", "but", "although", "instead", "for example", "for instance",
                "this means", "in other words", "shows that", "suggests that",
            ),
        )
        if not overlap and not has_explicit_signal:
            return None

        if self._contains_any(joined, ("not", "never", "no ", "without", "contrary", "however", "but")):
            return "CONTRADICTS"
        if self._contains_any(joined, ("because", "therefore", "thus", "result", "leads to", "causes", "due to")):
            return "CAUSES"
        if self._contains_any(joined, ("before", "after", "then", "next", "earlier", "later")):
            return "TEMPORAL_BEFORE"
        if self._contains_any(joined, ("for example", "for instance", "this means", "in other words")):
            return "ELABORATES"
        if self._contains_any(joined, ("shows that", "suggests that", "evidence", "demonstrates")):
            return "SUPPORTS"
        if len(overlap) >= 3:
            return "SUPPORTS"
        if len(overlap) >= 2:
            return "ASSOCIATED_WITH"
        return "ELABORATES" if has_explicit_signal else None

    def summarize_focus_nodes(self, labels: Sequence[Dict[str, Any]]) -> List[str]:
        counter = Counter()
        for item in labels:
            counter.update(item.get("concepts", []))
            counter.update(item.get("entities", []))
        return [name for name, _ in counter.most_common(8)]

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _extract_entities(self, text: str) -> List[str]:
        entities = set()
        for match in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b", text):
            if len(match) > 1:
                entities.add(match.strip())
        return sorted(entities)[:12]

    def _extract_concepts(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
        counter = Counter(token for token in tokens if token not in STOPWORDS)
        return [token for token, _ in counter.most_common(12)]

    def _labels_for_span(self, text: str) -> Dict[str, Any]:
        return {
            "text": text,
            "display_label": self._make_display_label("claim", text),
            "entities": self._extract_entities(text),
            "concepts": self._extract_concepts(text),
        }

    def _normalize_llm_bundle(self, payload: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        claims: List[str] = []
        claim_details: List[Dict[str, Any]] = []
        claim_label_map: Dict[str, str] = {}
        entity_label_map: Dict[str, str] = {}
        concept_label_map: Dict[str, str] = {}

        raw_claims = payload.get("claims") or []
        for item in raw_claims[:3]:
            if isinstance(item, str):
                claim_text = self._clean_text(item, max_chars=280)
                claim_entities: List[str] = []
                claim_concepts: List[str] = []
                display_label = ""
            elif isinstance(item, dict):
                claim_text = self._clean_text(item.get("text"), max_chars=280)
                claim_entities = self._normalize_name_list(item.get("entities"), limit=6)
                claim_concepts = self._normalize_name_list(item.get("concepts"), limit=6)
                display_label = self._clean_text(item.get("display_label"), max_chars=40)
            else:
                continue
            if not claim_text or claim_text in claim_label_map:
                continue
            final_label = self._normalize_display_label("claim", display_label) or self._make_display_label("claim", claim_text)
            claims.append(claim_text)
            claim_label_map[claim_text] = final_label
            claim_details.append(
                {
                    "text": claim_text,
                    "display_label": final_label,
                    "entities": claim_entities,
                    "concepts": claim_concepts,
                }
            )

        raw_entities = payload.get("entities") or []
        for item in raw_entities[:10]:
            text, label = self._normalize_named_item(item, "entity")
            if text and text not in entity_label_map:
                entity_label_map[text] = label

        raw_concepts = payload.get("concepts") or []
        for item in raw_concepts[:10]:
            text, label = self._normalize_named_item(item, "concept")
            if text and text not in concept_label_map:
                concept_label_map[text] = label

        for detail in claim_details:
            for entity in detail.get("entities", []):
                entity_label_map.setdefault(entity, self._make_display_label("entity", entity))
            for concept in detail.get("concepts", []):
                concept_label_map.setdefault(concept, self._make_display_label("concept", concept))

        if not claims:
            return self._extract_bundle_fallback(original_text)

        relations = self._normalize_relations(payload.get("relations"), claims)
        if not relations:
            relations = self._extract_internal_relations(claims)

        return {
            "claims": claims,
            "claim_details": claim_details,
            "entities": list(entity_label_map.keys())[:12],
            "concepts": list(concept_label_map.keys())[:12],
            "relations": relations,
            "display_labels": {
                "claims": claim_label_map,
                "entities": entity_label_map,
                "concepts": concept_label_map,
            },
        }

    def _normalize_named_item(self, item: Any, kind: str) -> tuple[str, str]:
        if isinstance(item, str):
            text = self._clean_text(item, max_chars=80)
            return text, self._make_display_label(kind, text)
        if isinstance(item, dict):
            text = self._clean_text(item.get("text"), max_chars=80)
            label = self._clean_text(item.get("display_label"), max_chars=32)
            return text, self._normalize_display_label(kind, label) or self._make_display_label(kind, text)
        return "", ""

    def _normalize_name_list(self, items: Any, limit: int) -> List[str]:
        if not isinstance(items, list):
            return []
        result: List[str] = []
        seen = set()
        for item in items[:limit]:
            text = self._clean_text(item, max_chars=80) if isinstance(item, str) else ""
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    def _normalize_relations(self, items: Any, claims: Sequence[str]) -> List[Dict[str, str]]:
        if not isinstance(items, list):
            return []
        valid_claims = set(claims)
        valid_types = {
            "SUPPORTS",
            "CONTRADICTS",
            "CAUSES",
            "TEMPORAL_BEFORE",
            "ELABORATES",
            "ASSOCIATED_WITH",
        }
        relations: List[Dict[str, str]] = []
        seen = set()
        for item in items[:6]:
            if not isinstance(item, dict):
                continue
            source_claim = self._clean_text(item.get("source_claim"), max_chars=280)
            target_claim = self._clean_text(item.get("target_claim"), max_chars=280)
            relation_type = self._clean_text(item.get("type"), max_chars=32).upper()
            if (
                not source_claim
                or not target_claim
                or source_claim not in valid_claims
                or target_claim not in valid_claims
                or relation_type not in valid_types
            ):
                continue
            key = (source_claim, target_claim, relation_type)
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "source_claim": source_claim,
                    "target_claim": target_claim,
                    "type": relation_type,
                }
            )
        return relations

    def _make_display_label(self, kind: str, text: str) -> str:
        cleaned = self._clean_text(text, max_chars=220)
        if not cleaned:
            return ""
        max_words = 6 if kind == "claim" else 3
        max_chars = 36 if kind == "claim" else 24
        words = cleaned.split()
        label = " ".join(words[:max_words])
        if len(label) > max_chars:
            label = label[: max_chars - 3].rstrip() + "..."
        elif len(words) > max_words or len(cleaned) > len(label):
            label = label.rstrip(" ,.;:") + "..."
        return label

    def _normalize_display_label(self, kind: str, label: str) -> str:
        cleaned = self._clean_text(label, max_chars=80)
        if not cleaned:
            return ""
        cleaned = re.sub(
            r"^(?:first|second|third|this|it|we|there|here|in this|for this)\b[\s,:-]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned:
            return ""
        max_words = 6 if kind == "claim" else 3
        max_chars = 36 if kind == "claim" else 24
        words = cleaned.split()
        shortened = " ".join(words[:max_words])
        if len(shortened) > max_chars:
            shortened = shortened[: max_chars - 3].rstrip() + "..."
        elif len(words) > max_words or len(cleaned) > len(shortened):
            shortened = shortened.rstrip(" ,.;:") + "..."
        return shortened

    def _clean_text(self, value: Any, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()
        if not text:
            return ""
        return text[:max_chars].strip()

    def _response_text(self, response: Any) -> str:
        try:
            content = ((response.choices or [None])[0].message.content or "")
        except Exception:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(getattr(item, "text", "") or ""))
            return "".join(parts).strip()
        return str(content).strip()

    def _extract_internal_relations(self, sentences: Iterable[str]) -> List[Dict[str, str]]:
        relations: List[Dict[str, str]] = []
        sentence_list = [sentence.strip() for sentence in sentences if sentence.strip()]
        for index in range(len(sentence_list) - 1):
            left = sentence_list[index]
            right = sentence_list[index + 1]
            joined = f"{left.lower()} {right.lower()}"
            left_labels = self._labels_for_span(left)
            right_labels = self._labels_for_span(right)
            overlap = (
                set(left_labels["concepts"]) | set(left_labels["entities"])
            ) & (
                set(right_labels["concepts"]) | set(right_labels["entities"])
            )
            relation_type = None
            if self._contains_any(joined, ("because", "therefore", "thus", "result", "causes", "leads to")):
                relation_type = "CAUSES"
            elif self._contains_any(joined, ("however", "but", "although", "instead")):
                relation_type = "CONTRADICTS"
            elif self._contains_any(joined, ("before", "after", "then", "next", "earlier", "later")):
                relation_type = "TEMPORAL_BEFORE"
            elif self._contains_any(joined, ("for example", "for instance", "this means", "in other words")):
                relation_type = "ELABORATES"
            elif len(overlap) >= 3:
                relation_type = "SUPPORTS"
            elif len(overlap) >= 1:
                relation_type = "ASSOCIATED_WITH"
            if not relation_type:
                continue
            relations.append(
                {
                    "source_claim": left[:240],
                    "target_claim": right[:240],
                    "type": relation_type,
                }
            )
        return relations

    def _contains_any(self, text: str, patterns: Sequence[str]) -> bool:
        return any(pattern in text for pattern in patterns)
