from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config.config import settings
from app.log.logger import get_model_logger
from app.service.client.api_client import GeminiApiClient
from app.service.model.model_aliases import (
    ModelAliasResolutionError,
    get_base_model_name,
    normalize_model_aliases,
    resolve_model_alias,
    split_model_suffixes,
)

logger = get_model_logger()


@dataclass(frozen=True)
class ResolvedModel:
    public_name: str
    upstream_name: str

    @property
    def is_alias(self) -> bool:
        return self.public_name != self.upstream_name


class ModelService:
    def get_model_aliases(self) -> Dict[str, str]:
        return normalize_model_aliases(settings.MODEL_ALIASES)

    def resolve_model_alias(self, model: str) -> str:
        return resolve_model_alias(model, self.get_model_aliases())

    def resolve_request_model(self, model: str) -> ResolvedModel:
        public_name = (model or "").strip()
        return ResolvedModel(
            public_name=public_name,
            upstream_name=self.resolve_model_alias(public_name),
        )

    def is_model_supported(self, model: str) -> bool:
        if not model or not isinstance(model, str):
            return False

        try:
            resolved_model = self.resolve_model_alias(model)
        except ModelAliasResolutionError as error:
            logger.warning(f"Invalid MODEL_ALIASES configuration for '{model}': {error}")
            return False

        base_model, suffixes = split_model_suffixes(resolved_model)
        if not base_model or base_model in settings.FILTERED_MODELS:
            return False

        if not suffixes:
            return True

        for suffix in suffixes:
            if suffix == "-search" and base_model not in settings.SEARCH_MODELS:
                return False
            if (
                suffix in {"-image", "-image-generation"}
                and base_model not in settings.IMAGE_MODELS
            ):
                return False
            if (
                suffix == "-non-thinking"
                and base_model not in settings.THINKING_MODELS
            ):
                return False
            if suffix == "-chat" and base_model != settings.CREATE_IMAGE_MODEL:
                return False

        return True

    def is_image_chat_model(self, model: str) -> bool:
        try:
            return (
                self.resolve_model_alias(model)
                == f"{settings.CREATE_IMAGE_MODEL}-chat"
            )
        except ModelAliasResolutionError:
            return False

    async def get_gemini_models(self, api_key: str) -> Optional[Dict[str, Any]]:
        api_client = GeminiApiClient(base_url=settings.BASE_URL)
        gemini_models = await api_client.get_models(api_key)

        if gemini_models is None:
            logger.error("从 API 客户端获取模型列表失败。")
            return None

        try:
            filtered_models_list = []
            for model in gemini_models.get("models", []):
                model_id = model["name"].split("/")[-1]
                if model_id not in settings.FILTERED_MODELS:
                    filtered_models_list.append(model)
                else:
                    logger.debug(f"Filtered out model: {model_id}")

            gemini_models["models"] = filtered_models_list
            return gemini_models
        except Exception as e:
            logger.error(f"处理模型列表时出错: {e}")
            return None

    def _append_base_derived_models(self, models_json: Dict[str, Any]) -> None:
        model_mapping = {
            x.get("name", "").split("/", maxsplit=1)[-1]: x
            for x in models_json.get("models", [])
        }

        def add_derived_model(base_name: str, suffix: str, display_suffix: str) -> None:
            model = model_mapping.get(base_name)
            if not model:
                logger.warning(
                    f"Base model '{base_name}' not found for derived model '{suffix}'."
                )
                return
            item = deepcopy(model)
            item["name"] = f"models/{base_name}{suffix}"
            display_name = f'{item.get("displayName", base_name)}{display_suffix}'
            item["displayName"] = display_name
            item["description"] = display_name
            models_json["models"].append(item)

        if settings.SEARCH_MODELS:
            for name in settings.SEARCH_MODELS:
                add_derived_model(name, "-search", " For Search")
        if settings.IMAGE_MODELS:
            for name in settings.IMAGE_MODELS:
                add_derived_model(name, "-image", " For Image")
        if settings.THINKING_MODELS:
            for name in settings.THINKING_MODELS:
                add_derived_model(name, "-non-thinking", " Non Thinking")

    def _append_gemini_alias_models(self, models_json: Dict[str, Any]) -> None:
        alias_map = self.get_model_aliases()
        if not alias_map:
            return

        model_mapping = {
            x.get("name", "").split("/", maxsplit=1)[-1]: x
            for x in models_json.get("models", [])
        }
        existing_names = set(model_mapping.keys())

        def append_alias_entry(
            public_name: str, target_name: str, description: str
        ) -> None:
            if public_name in existing_names:
                return

            base_target = get_base_model_name(target_name)
            base_model = model_mapping.get(base_target)
            if not base_model:
                logger.warning(
                    f"Skipping alias '{public_name}' because base model '{base_target}' is unavailable in the upstream model list."
                )
                return

            item = deepcopy(base_model)
            item["name"] = f"models/{public_name}"
            item["displayName"] = public_name
            item["description"] = description
            models_json["models"].append(item)
            existing_names.add(public_name)

        for alias_name in alias_map:
            alias_base_name, alias_suffixes = split_model_suffixes(alias_name)
            try:
                resolved_target = self.resolve_model_alias(alias_name)
            except ModelAliasResolutionError as error:
                logger.warning(f"Skipping alias '{alias_name}': {error}")
                continue

            if not self.is_model_supported(alias_name):
                logger.warning(
                    f"Skipping alias '{alias_name}' because its target '{resolved_target}' is not supported."
                )
                continue

            description = f"Alias for {resolved_target}"
            append_alias_entry(alias_name, resolved_target, description)

            _, target_suffixes = split_model_suffixes(resolved_target)
            if alias_suffixes or target_suffixes:
                continue

            base_target = get_base_model_name(resolved_target)
            if base_target in settings.SEARCH_MODELS:
                append_alias_entry(
                    f"{alias_base_name}-search",
                    f"{base_target}-search",
                    f"Alias for {base_target}-search",
                )
            if base_target in settings.IMAGE_MODELS:
                append_alias_entry(
                    f"{alias_base_name}-image",
                    f"{base_target}-image",
                    f"Alias for {base_target}-image",
                )
            if base_target in settings.THINKING_MODELS:
                append_alias_entry(
                    f"{alias_base_name}-non-thinking",
                    f"{base_target}-non-thinking",
                    f"Alias for {base_target}-non-thinking",
                )

    def build_gemini_public_models(
        self, gemini_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        models_json = deepcopy(gemini_models)
        self._append_base_derived_models(models_json)
        self._append_gemini_alias_models(models_json)
        return models_json

    async def get_gemini_openai_models(self, api_key: str) -> Optional[Dict[str, Any]]:
        """获取 Gemini 模型并转换为 OpenAI 格式"""
        gemini_models = await self.get_gemini_models(api_key)
        if gemini_models is None:
            return None
        
        return await self.convert_to_openai_models_format(gemini_models)

    async def convert_to_openai_models_format(
        self, gemini_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        openai_format = {"object": "list", "data": [], "success": True}
        model_mapping = {
            model["name"].split("/")[-1]: model for model in gemini_models.get("models", [])
        }
        existing_ids = set()
        created_at = int(datetime.now(timezone.utc).timestamp())

        def add_openai_model(public_name: str, target_name: str) -> None:
            if public_name in existing_ids:
                return

            base_target = get_base_model_name(target_name)
            base_model = model_mapping.get(base_target)
            root_name = (
                base_model["name"] if base_model else f"models/{base_target}"
            )
            openai_model = {
                "id": public_name,
                "object": "model",
                "created": created_at,
                "owned_by": "google",
                "permission": [],
                "root": root_name,
                "parent": None,
            }
            openai_format["data"].append(openai_model)
            existing_ids.add(public_name)

        for model_id in model_mapping:
            add_openai_model(model_id, model_id)
            if model_id in settings.SEARCH_MODELS:
                add_openai_model(f"{model_id}-search", f"{model_id}-search")
            if model_id in settings.IMAGE_MODELS:
                add_openai_model(f"{model_id}-image", f"{model_id}-image")
            if model_id in settings.THINKING_MODELS:
                add_openai_model(
                    f"{model_id}-non-thinking", f"{model_id}-non-thinking"
                )

        if settings.CREATE_IMAGE_MODEL:
            add_openai_model(
                f"{settings.CREATE_IMAGE_MODEL}-chat",
                f"{settings.CREATE_IMAGE_MODEL}-chat",
            )

        for alias_name in self.get_model_aliases():
            alias_base_name, alias_suffixes = split_model_suffixes(alias_name)
            try:
                resolved_target = self.resolve_model_alias(alias_name)
            except ModelAliasResolutionError as error:
                logger.warning(f"Skipping alias '{alias_name}': {error}")
                continue

            if not self.is_model_supported(alias_name):
                logger.warning(
                    f"Skipping alias '{alias_name}' because its target '{resolved_target}' is not supported."
                )
                continue

            add_openai_model(alias_name, resolved_target)

            _, target_suffixes = split_model_suffixes(resolved_target)
            if alias_suffixes or target_suffixes:
                continue

            base_target = get_base_model_name(resolved_target)
            if base_target in settings.SEARCH_MODELS:
                add_openai_model(
                    f"{alias_base_name}-search", f"{base_target}-search"
                )
            if base_target in settings.IMAGE_MODELS:
                add_openai_model(f"{alias_base_name}-image", f"{base_target}-image")
            if base_target in settings.THINKING_MODELS:
                add_openai_model(
                    f"{alias_base_name}-non-thinking",
                    f"{base_target}-non-thinking",
                )
            if base_target == settings.CREATE_IMAGE_MODEL:
                add_openai_model(
                    f"{alias_base_name}-chat", f"{base_target}-chat"
                )

        return openai_format

    async def check_model_support(self, model: str) -> bool:
        return self.is_model_supported(model)
