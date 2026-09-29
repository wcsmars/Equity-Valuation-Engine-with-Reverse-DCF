"""Offline checks for local settings persistence and browser access."""

import builtins
import runpy
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import app as backend
from backend import ai_service


class AIConfigurationTests(unittest.TestCase):
    def test_missing_key_does_not_import_sdk(self):
        with (
            patch.dict(ai_service.os.environ, {}, clear=True),
            patch("builtins.__import__", side_effect=AssertionError("Unexpected import")),
        ):
            with self.assertRaises(ai_service.AIError):
                ai_service._client()

    def test_model_override_loads_without_importing_sdk(self):
        original_import = builtins.__import__

        def without_sdk(name, *args, **kwargs):
            if name == "anthropic" or name.startswith("anthropic."):
                raise AssertionError("SDK should load only when an AI client is requested")
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(ai_service.os.environ, {"ANTHROPIC_MODEL": "custom-model"}, clear=True),
            patch("builtins.__import__", side_effect=without_sdk),
        ):
            namespace = runpy.run_path(ai_service.__file__)
        self.assertEqual(namespace["MODEL"], "custom-model")


class SettingsTests(unittest.TestCase):
    def test_safe_keys_are_saved_and_applied(self):
        request = backend.SettingsRequest(
            anthropic_api_key="  sk-ant-example_123.token  ",
            fmp_api_key="fmp-example_456",
        )
        with (
            patch.object(backend, "_upsert_env_file") as save,
            patch.dict(backend.os.environ, {}, clear=True),
            patch.object(backend, "_fmp"),
        ):
            result = backend.settings(request)
            expected = {
                "ANTHROPIC_API_KEY": "sk-ant-example_123.token",
                "FMP_API_KEY": "fmp-example_456",
            }
            save.assert_called_once_with(expected)
            for key, value in expected.items():
                self.assertEqual(backend.os.environ[key], value)
            self.assertTrue(result["anthropic_enabled"])
            self.assertTrue(result["fmp_enabled"])

    def test_unsafe_keys_do_not_write_or_change_environment(self):
        unsafe = [
            "sk-ant-$(touch unwanted)",
            "sk-ant-`whoami`",
            "sk-ant-example;echo unwanted",
            "sk-ant-example\nOTHER=value",
            'sk-ant-"quoted"',
            "sk ant example",
        ]
        for field in ("anthropic_api_key", "fmp_api_key"):
            for value in unsafe:
                with self.subTest(field=field, value=value):
                    values = {
                        "anthropic_api_key": "sk-ant-valid",
                        "fmp_api_key": "fmp-valid",
                        field: value,
                    }
                    with (
                        patch.object(backend, "_upsert_env_file") as save,
                        patch.dict(backend.os.environ, {"EXISTING": "value"}, clear=True),
                    ):
                        with self.assertRaises(HTTPException) as caught:
                            backend.settings(backend.SettingsRequest(**values))
                        self.assertEqual(caught.exception.status_code, 400)
                        self.assertNotIn(value, caught.exception.detail)
                        save.assert_not_called()
                        self.assertEqual(dict(backend.os.environ), {"EXISTING": "value"})

    def test_empty_settings_do_not_write(self):
        with patch.object(backend, "_upsert_env_file") as save:
            with self.assertRaises(HTTPException) as caught:
                backend.settings(backend.SettingsRequest(anthropic_api_key="   "))
            self.assertEqual(caught.exception.status_code, 400)
            save.assert_not_called()


class CorsTests(unittest.TestCase):
    def test_dynamic_local_origins_allowed_and_external_origins_rejected(self):
        middleware = next(
            item for item in backend.app.user_middleware
            if item.cls is CORSMiddleware
        )
        cors = middleware.cls(backend.app, *middleware.args, **middleware.kwargs)
        for origin in (
            "http://localhost:3000",
            "http://127.0.0.1:49152",
            "http://localhost",
            "https://localhost:3000",
            "http://[::1]:3000",
        ):
            with self.subTest(origin=origin):
                self.assertTrue(cors.is_allowed_origin(origin))
        for origin in (
            "https://example.com",
            "http://localhost.evil.example:3000",
            "http://127.0.0.1.evil.example:3000",
            "http://192.168.1.2:3000",
            "null",
        ):
            with self.subTest(origin=origin):
                self.assertFalse(cors.is_allowed_origin(origin))


if __name__ == "__main__":
    unittest.main()
