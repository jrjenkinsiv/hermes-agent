"""Memory providers use the same scoped, permission-checked MCP client as other plugins."""

import json
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


PROVIDER = '''
import json
from agent.memory_provider import MemoryProvider

class Probe(MemoryProvider):
    def __init__(self, ctx): self.ctx = ctx
    @property
    def name(self): return "contextprobe"
    def is_available(self): return True
    def initialize(self, session_id, **kwargs): pass
    def get_tool_schemas(self): return []
    def prefetch(self, query, *, session_id=""):
        return json.dumps(self.ctx.call_mcp("memory-server", "recall", {"query": query}, timeout=2))

def register(ctx):
    ctx.register_memory_provider(Probe(ctx))
'''


def _home(tmp_path, name, isolation, *, granted):
    home = tmp_path / name
    directory = home / "plugins" / "contextprobe"
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(PROVIDER, encoding="utf-8")
    (directory / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "contextprobe", "version": "1.0"}), encoding="utf-8")
    config = {
        "memory": {"provider": "contextprobe", "memory_enabled": False},
        "plugins": {"isolation": isolation, "entries": {
            "contextprobe": {"mcp_allowlist": ["memory-server"] if granted else []},
        }},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return home


def _fake_mcp(monkeypatch, calls):
    from tools import mcp_tool_handlers

    def factory(server, tool, timeout):
        def call(arguments):
            calls.append((str(get_hermes_home()), server, tool, arguments, timeout))
            return json.dumps({"result": {"query": arguments["query"], "home": str(get_hermes_home())}})
        return call

    monkeypatch.setattr(mcp_tool_handlers, "_make_tool_handler", factory)


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_directory_memory_provider_calls_existing_mcp_and_keeps_profile_grants(tmp_path, monkeypatch, isolation):
    from plugins.memory import load_memory_provider

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = []
    _fake_mcp(monkeypatch, calls)
    homes = [_home(tmp_path, "granted", isolation, granted=True),
             _home(tmp_path, "denied", isolation, granted=False)]
    managers = []
    try:
        for home in homes:
            token = set_hermes_home_override(str(home))
            try:
                provider = load_memory_provider("contextprobe")
                assert provider is not None
                managers.append(plugins_mod.get_plugin_manager())
                if home == homes[0]:
                    result = json.loads(provider.prefetch("existing preference"))
                    assert result == {"ok": True, "result": {
                        "query": "existing preference", "home": str(home)}}
                else:
                    with pytest.raises(Exception, match="not allowed to call MCP"):
                        provider.prefetch("must not reach the server")
            finally:
                reset_hermes_home_override(token)
        assert calls == [(str(homes[0]), "memory-server", "recall", {"query": "existing preference"}, 2.0)]
    finally:
        for manager in managers:
            manager.unload()
            if isolation == "host":
                manager._plugin_host().shutdown()


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_memory_provider_preserves_native_mcp_error_envelope(tmp_path, monkeypatch, isolation):
    from plugins.memory import load_memory_provider
    from tools import mcp_tool_handlers

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(mcp_tool_handlers, "_make_tool_handler", lambda *args: lambda arguments: json.dumps(
        {"error": "memory unavailable"}))
    home = _home(tmp_path, "failure", isolation, granted=True)
    token = set_hermes_home_override(str(home))
    manager = None
    try:
        provider = load_memory_provider("contextprobe")
        manager = plugins_mod.get_plugin_manager()
        assert json.loads(provider.prefetch("preference")) == {"ok": False, "error": "memory unavailable"}
    finally:
        if manager is not None:
            manager.unload()
            if isolation == "host":
                manager._plugin_host().shutdown()
        reset_hermes_home_override(token)


def test_memory_collector_does_not_expose_arbitrary_plugin_context_attributes():
    from plugins.memory import _ProviderCollector

    with pytest.raises(AttributeError):
        _ProviderCollector("contextprobe").not_a_context_method()
