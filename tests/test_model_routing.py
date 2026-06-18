import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("DATABASE_TYPE", "sqlite")

try:
    from google import genai as _google_genai  # type: ignore
except Exception:
    google_module = sys.modules.setdefault("google", types.ModuleType("google"))
    genai_module = types.ModuleType("genai")
    genai_module.Client = object
    google_module.genai = genai_module
    sys.modules["google.genai"] = genai_module

from app.domain.gemini_models import GeminiContent, GeminiRequest
from app.domain.openai_models import ChatRequest
from app.router import gemini_routes, openai_routes, vertex_express_routes
from app.service.model.model_service import ResolvedModel


class ModelRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_chat_route_resolves_alias_before_upstream_call(self):
        request = ChatRequest(
            model="fast",
            messages=[{"role": "user", "content": "hi"}],
        )
        chat_service = Mock()
        chat_service.create_chat_completion = AsyncMock(
            return_value={"model": "fast", "choices": []}
        )
        chat_service.create_image_chat_completion = AsyncMock()
        key_manager = Mock()
        key_manager.get_paid_key = AsyncMock(return_value="paid-key")

        with (
            patch.object(
                openai_routes.model_service,
                "check_model_support",
                new=AsyncMock(return_value=True),
            ),
            patch.object(
                openai_routes.model_service,
                "resolve_request_model",
                return_value=ResolvedModel("fast", "gemini-2.5-flash"),
            ),
            patch.object(
                openai_routes.model_service,
                "is_image_chat_model",
                return_value=False,
            ),
        ):
            response = await openai_routes.chat_completion(
                request=request,
                allowed_token="token",
                api_key="api-key",
                key_manager=key_manager,
                chat_service=chat_service,
            )

        self.assertEqual(response["model"], "fast")
        routed_request = chat_service.create_chat_completion.await_args.args[0]
        self.assertEqual(routed_request.model, "gemini-2.5-flash")
        self.assertEqual(
            chat_service.create_chat_completion.await_args.kwargs["public_model"],
            "fast",
        )

    async def test_gemini_route_resolves_alias_before_upstream_call(self):
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "hi"}])])
        chat_service = Mock()
        chat_service.generate_content = AsyncMock(return_value={"model": "fast"})

        with (
            patch.object(
                gemini_routes.model_service,
                "check_model_support",
                new=AsyncMock(return_value=True),
            ),
            patch.object(
                gemini_routes.model_service,
                "resolve_request_model",
                return_value=ResolvedModel("fast", "gemini-2.5-flash"),
            ),
        ):
            response = await gemini_routes.generate_content(
                model_name="fast",
                request=request,
                allowed_token="token",
                api_key="api-key",
                key_manager=Mock(),
                chat_service=chat_service,
            )

        self.assertEqual(response["model"], "fast")
        self.assertEqual(
            chat_service.generate_content.await_args.kwargs["model"],
            "gemini-2.5-flash",
        )
        self.assertEqual(
            chat_service.generate_content.await_args.kwargs["public_model"],
            "fast",
        )

    async def test_vertex_route_resolves_alias_before_upstream_call(self):
        request = GeminiRequest(contents=[GeminiContent(role="user", parts=[{"text": "hi"}])])
        chat_service = Mock()
        chat_service.generate_content = AsyncMock(return_value={"model": "fast"})

        with (
            patch.object(
                vertex_express_routes.model_service,
                "check_model_support",
                new=AsyncMock(return_value=True),
            ),
            patch.object(
                vertex_express_routes.model_service,
                "resolve_request_model",
                return_value=ResolvedModel("fast", "gemini-2.5-flash"),
            ),
        ):
            response = await vertex_express_routes.generate_content(
                model_name="fast",
                request=request,
                allowed_token="token",
                api_key="api-key",
                key_manager=Mock(),
                chat_service=chat_service,
            )

        self.assertEqual(response["model"], "fast")
        self.assertEqual(
            chat_service.generate_content.await_args.kwargs["model"],
            "gemini-2.5-flash",
        )
        self.assertEqual(
            chat_service.generate_content.await_args.kwargs["public_model"],
            "fast",
        )

    async def test_vertex_list_models_uses_vertex_key_and_source_aware_service(self):
        key_manager = Mock()
        key_manager.get_next_working_vertex_key = AsyncMock(return_value="vertex-key")
        key_manager.get_random_valid_key = AsyncMock(return_value="gemini-key")
        base_models = {
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "displayName": "Gemini 2.5 Flash",
                    "description": "flash",
                }
            ]
        }
        mocked_public_models = AsyncMock(return_value=base_models)

        with (
            patch.object(
                vertex_express_routes.model_service,
                "get_public_source_models",
                mocked_public_models,
            ),
            patch.object(
                vertex_express_routes.model_service,
                "build_gemini_public_models",
                return_value=base_models,
            ),
        ):
            response = await vertex_express_routes.list_models(
                allowed_token="token",
                key_manager=key_manager,
            )

        self.assertEqual(response, base_models)
        key_manager.get_next_working_vertex_key.assert_awaited_once()
        key_manager.get_random_valid_key.assert_not_awaited()
        mocked_public_models.assert_awaited_once_with("vertex", "vertex-key")
