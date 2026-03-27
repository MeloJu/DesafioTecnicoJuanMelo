"""
Shared pytest fixtures for all test layers.

Unit tests: mock all external dependencies (LLM, vision models, disk I/O).
Component tests: real module internals, external deps still mocked.
Integration tests: two or more real modules, LLM optionally mocked.
E2E tests: full pipeline with real Llama and CLIP — mark with @pytest.mark.e2e.
"""
