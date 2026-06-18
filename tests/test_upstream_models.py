import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_TYPE", "sqlite")

from app.config.config import settings
from app.router import model_routes
from app.service.model.model_service import ModelService


class UpstreamModelServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_upstream_models_gemini_follows_next_page_token(self):
        service = ModelService()
        first_page = {
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "displayName": "Gemini 2.5 Flash",
                    "description": "flash",
                }
            ],
            "nextPageToken": "page-2",
        }
        second_page = {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "description": "pro",
                }
            ]
        }
        mocked_get_models = AsyncMock(side_effect=[first_page, second_page])

        with patch(
            "app.service.model.model_service.GeminiApiClient.get_models",
            mocked_get_models,
        ):
            models = await service.get_upstream_models("gemini", "gemini-key")

        self.assertEqual([model["id"] for model in models], ["gemini-2.5-flash", "gemini-2.5-pro"])
        self.assertEqual(models[0]["raw_name"], "models/gemini-2.5-flash")
        self.assertEqual(models[1]["display_name"], "Gemini 2.5 Pro")
        self.assertEqual(mocked_get_models.await_count, 2)
        self.assertEqual(mocked_get_models.await_args_list[0].args, ("gemini-key",))
        self.assertIsNone(mocked_get_models.await_args_list[0].kwargs["page_token"])
        self.assertEqual(
            mocked_get_models.await_args_list[1].kwargs["page_token"], "page-2"
        )

    async def test_get_upstream_models_openai_normalizes_data_payload(self):
        service = ModelService()
        mocked_get_models = AsyncMock(
            return_value={
                "data": [
                    {"id": "gpt-4.1", "description": "OpenAI model"},
                ]
            }
        )

        with patch(
            "app.service.model.model_service.OpenaiApiClient.get_models",
            mocked_get_models,
        ):
            models = await service.get_upstream_models("openai", "openai-key")

        self.assertEqual(
            models,
            [
                {
                    "id": "gpt-4.1",
                    "raw_name": "gpt-4.1",
                    "display_name": "gpt-4.1",
                    "description": "OpenAI model",
                    "raw": {"id": "gpt-4.1", "description": "OpenAI model"},
                }
            ],
        )


class UpstreamModelRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(model_routes.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_upstream_route_requires_bearer_token_and_returns_models(self):
        key_manager = Mock()
        key_manager.get_random_valid_key = AsyncMock(return_value="gemini-key")
        key_manager.get_next_working_vertex_key = AsyncMock(return_value="vertex-key")
        self.app.dependency_overrides[model_routes.get_key_manager] = lambda: key_manager

        mocked_models = AsyncMock(
            return_value=[
                {
                    "id": "gemini-2.5-flash",
                    "raw_name": "models/gemini-2.5-flash",
                    "display_name": "Gemini 2.5 Flash",
                    "description": "flash",
                    "raw": {"name": "models/gemini-2.5-flash"},
                }
            ]
        )

        with (
            patch.object(settings, "ALLOWED_TOKENS", ["mgmt-token"]),
            patch.object(settings, "AUTH_TOKEN", "mgmt-token"),
            patch.object(model_routes.model_service, "get_upstream_models", mocked_models),
        ):
            unauthorized = self.client.get("/api/models/upstream?source=gemini")
            response = self.client.get(
                "/api/models/upstream?source=gemini",
                headers={"Authorization": "Bearer mgmt-token"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "source": "gemini",
                "models": [
                    {
                        "id": "gemini-2.5-flash",
                        "raw_name": "models/gemini-2.5-flash",
                        "display_name": "Gemini 2.5 Flash",
                        "description": "flash",
                        "raw": {"name": "models/gemini-2.5-flash"},
                    }
                ],
            },
        )
        key_manager.get_random_valid_key.assert_awaited_once()
        mocked_models.assert_awaited_once_with("gemini", "gemini-key")
