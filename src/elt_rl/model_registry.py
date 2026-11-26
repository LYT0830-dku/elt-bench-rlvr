"""Model metadata, Cookbook tokenizers/renderers and model-specific tool codecs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from elt_rl.registry import Registry, UnsupportedCapability

CATALOG_PATH = Path(__file__).with_name("model_catalog.json")
CATALOG_FROZEN_SHA256 = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def catalog():
    data = json.loads(CATALOG_PATH.read_text())
    if data.get("format") != "elt_model_catalog_v1":
        raise ValueError("Unsupported model catalog")
    return data


DEFAULT_MODEL = catalog()["default_model"]


class TMLToolCodec:
    def initial_messages(self, observation, renderer):
        from elt_rl.environment import initial_messages

        return initial_messages(observation)

    def parse_action(self, message, termination, text):
        from elt_rl.action_protocol import native_action

        return native_action(message, termination, text)

    def observations(self, message, observation):
        from elt_rl.action_protocol import observation_messages

        return observation_messages(message, observation)


class CookbookToolCodec(TMLToolCodec):
    def initial_messages(self, observation, renderer):
        from elt_rl.action_protocol import declared_tools

        functions = [
            tool["function"]
            for tool in declared_tools(loading=observation.get("execution_backend") == "native")
        ]
        prefix = renderer.create_conversation_prefix_with_tools(
            functions, system_prompt=observation["instructions"]
        )
        return prefix + [
            {
                "role": "user",
                "content": json.dumps(
                    {key: value for key, value in observation.items() if key != "instructions"},
                    default=str,
                ),
            }
        ]


TOOL_CODECS = Registry("model tool codec")
TOOL_CODECS.register("tml_native", TMLToolCodec())
TOOL_CODECS.register("cookbook_native", CookbookToolCodec())


@dataclass(frozen=True)
class ModelAdapter:
    configuration: dict

    @property
    def id(self):
        return self.configuration["id"]

    @property
    def codec(self):
        return TOOL_CODECS.get(self.configuration["tool_codec"])

    def validate(self, *, renderer=None, context_tokens=None, effort=None, training=False):
        for capability in (
            "sampling",
            "tool_calling",
            "logprobs",
            *(("lora_training",) if training else ()),
        ):
            if self.configuration.get(capability) is not True:
                raise UnsupportedCapability(
                    f"Model {self.id} lacks required capability: {capability}"
                )
        if renderer is not None and renderer not in self.configuration["renderers"]:
            raise UnsupportedCapability(
                f"Native tool calls require the {' or '.join(self.configuration['renderers'])} renderer for model {self.id}; unsupported renderer {renderer}"
            )
        if effort is not None and not self.configuration["effort_conditioning"]:
            raise UnsupportedCapability(
                "This model's Cookbook renderer does not expose numeric effort conditioning"
            )
        if context_tokens is not None and context_tokens > self.configuration["context_limit"]:
            raise UnsupportedCapability("Requested context exceeds verified model contract")
        TOOL_CODECS.get(self.configuration["tool_codec"])
        return self

    def renderer_name(self, requested=None):
        self.validate(renderer=requested)
        return requested or self.configuration["renderers"][0]

    def renderer(self, requested=None, effort=None):
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        from elt_rl.native_renderers import register_native_renderers

        self.validate(renderer=requested, effort=effort)
        name = self.renderer_name(requested)
        tokenizer = get_tokenizer(self.id)
        register_native_renderers()
        renderer = renderers.get_renderer(name, tokenizer)
        return name, renderer

    def fingerprint(self):
        return {
            "model": self.id,
            "profile": self.configuration,
            "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        }

    def validate_checkpoint(self, checkpoint):
        if checkpoint.get("config", {}).get("model") != self.id:
            raise UnsupportedCapability("Checkpoint belongs to another model")
        expected = checkpoint.get("model_adapter")
        if expected is not None and expected != self.fingerprint():
            raise UnsupportedCapability("Checkpoint model/renderer contract changed")


class ModelRegistry(Registry[ModelAdapter]):
    def __init__(self):
        super().__init__("model")


MODELS = ModelRegistry()
for _profile in catalog()["models"]:
    MODELS.register(_profile["id"], ModelAdapter(deepcopy(_profile)))
