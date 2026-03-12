# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import erdantic as erd
from pydantic import BaseModel

import aind_behavior_dynamic_foraging
import aind_behavior_dynamic_foraging.task_logic

SOURCE_ROOT = "https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging/tree/main/src/"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AIND Dynamic Foraging"
copyright = "2026, Allen Institute for Neural Dynamics"
release = aind_behavior_dynamic_foraging.__semver__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx-jsonschema",
    "sphinx_jinja",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True

autodoc_pydantic_model_show_json = False
autodoc_pydantic_settings_show_json = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Aind Dynamic Foraging"
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "light_logo": "light-logo.svg",
    "dark_logo": "dark-logo.svg",
}

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False


# -- Options for linkcode extension ---------------------------------------


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"{SOURCE_ROOT}/{filename}.py"


# -- Class diagram generation

_static_path = "_static"


def export_model_diagram(model: BaseModel, root: str = _static_path) -> None:
    diagram = erd.create(model)
    diagram.draw(f"{root}/{model.__name__}.svg")


# -- Dataset rendering

with open(f"{_static_path}/dataset.html", "w", encoding="utf-8") as f:
    from aind_behavior_dynamic_foraging.data_contract import render_dataset

    f.write(render_dataset(version=release))


def setup(app):
    import typing

    from erdantic.plugins import register_plugin
    from erdantic.plugins.pydantic import get_fields_from_pydantic_model, is_pydantic_model

    import aind_behavior_dynamic_foraging.task_logic.trial_generators.composite_trial_generator as _comp_mod

    _EXCLUDED_FIELDS = {"model_config"}

    def _get_fields_filtered(model):
        return [f for f in get_fields_from_pydantic_model(model) if f.name not in _EXCLUDED_FIELDS]

    register_plugin(key="pydantic", predicate_fn=is_pydantic_model, get_fields_fn=_get_fields_filtered)
    from aind_behavior_dynamic_foraging.task_logic import (
        AindDynamicForagingTaskLogic,
        AindDynamicForagingTaskParameters,
        TrialGeneratorSpec,
    )
    from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

    # erdantic calls model_rebuild(force=True) without a types namespace, so pydantic tries to
    # resolve ForwardRef("TrialGeneratorSpec") in composite_trial_generator's module globals,
    # where it isn't defined (circular import prevents it). Injecting it here fixes the lookup.
    _comp_mod.TrialGeneratorSpec = TrialGeneratorSpec
    AindDynamicForagingTaskParameters.model_rebuild(_types_namespace={"TrialGeneratorSpec": TrialGeneratorSpec})

    # Task-level diagrams
    export_model_diagram(AindDynamicForagingTaskLogic, _static_path)
    export_model_diagram(TrialOutcome, _static_path)

    # There may be an easier way to render the block generators, but for now we introspect the union type
    _union_members = typing.get_args(typing.get_args(TrialGeneratorSpec.__value__)[0])
    _generator_models = []
    for _m in _union_members:
        _origin = typing.get_origin(_m)
        _model = _origin if _origin is not None else _m
        if is_pydantic_model(_model) and _model not in _generator_models:
            _generator_models.append(_model)

    # One SVG per generator spec
    for _model in _generator_models:
        export_model_diagram(_model, _static_path)

    # Combined diagram covering all generators
    erd.create(*_generator_models).draw(f"{_static_path}/TrialGeneratorSpec.svg")
