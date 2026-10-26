# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
import argparse
import ast
import gc
import io
import json
import logging
import os
import shutil
import sys
import tokenize
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projects.holobrain.utils import load_checkpoint, load_config  # noqa: E402
from robo_orchard_lab.models.holobrain import HoloBrainProcessor  # noqa: E402
from robo_orchard_lab.models.holobrain.pipeline import (  # noqa: E402
    HoloBrainInferencePipeline,
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402
from robo_orchard_lab.utils import log_basic_config  # noqa: E402

__all__ = ["main"]

logger = logging.getLogger(__file__)


def main(args: argparse.Namespace) -> None:
    """Export using local resources, without modifying training artifacts.

    ``--kwargs`` accepts ``vlm_pretrain``, ``checkpoint`` and an optional
    ``urdf`` file or directory. Otherwise URDF assets come from the training
    run's ``urdf`` directory next to ``configs``. Configuration copies are
    relocated before import because some configs read URDFs at import time.
    """
    kwargs = {}
    if args.kwargs is not None:
        if os.path.isfile(args.kwargs):
            with open(args.kwargs) as fh:
                kwargs = json.load(fh)
        else:
            kwargs = json.loads(args.kwargs)
    source = Path(args.config).resolve()
    workspace = Path(args.workspace).resolve()
    config_dir = workspace / "configs"
    if config_dir == source.parent:
        raise ValueError(
            "Export workspace must not overwrite training configs"
        )
    urdf_source = Path(
        kwargs.pop("urdf", None) or source.parent.parent / "urdf"
    ).resolve()
    if not urdf_source.exists():
        raise FileNotFoundError(f"URDF resource does not exist: {urdf_source}")
    asset_source = urdf_source if urdf_source.is_dir() else urdf_source.parent
    asset_dir = workspace / "model" / "urdf"
    if asset_source == asset_dir:
        raise ValueError("Export URDF directory must differ from its source")
    shutil.copytree(asset_source, asset_dir, dirs_exist_ok=True)
    urdf_target = (
        asset_dir if urdf_source.is_dir() else asset_dir / urdf_source.name
    )
    shutil.copytree(
        source.parent,
        config_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # Preserve comments and formatting while rebinding literal URDF paths.
    # Other registered datasets may reference assets unused by this export.
    for config_file in config_dir.rglob("*.py"):
        tokens = []
        for token in tokenize.generate_tokens(
            io.StringIO(config_file.read_text()).readline
        ):
            if token.type == tokenize.STRING and token.string.lstrip(
                "rRuUbB"
            ).startswith(("'", '"')):
                value = ast.literal_eval(token.string)
                if isinstance(value, str) and value.endswith(".urdf"):
                    try:
                        replacement = _relocate_paths(
                            {"urdf": value}, None, urdf_target
                        )["urdf"]
                    except FileNotFoundError:
                        replacement = value
                    token = token._replace(string=repr(replacement))
            tokens.append(token)
        config_file.write_text(tokenize.untokenize(tokens))

    module = load_config(str(config_dir / source.name))
    build_model = module.build_model
    build_processors = module.build_processors
    config = dict(module.config)
    config.update(kwargs)
    vlm = config.get("vlm_pretrain")
    if vlm is not None and Path(vlm).exists():
        vlm = str(Path(vlm).resolve())
    config = _relocate_paths(config, vlm, urdf_target)
    logger.info("\n" + json.dumps(config, indent=4))

    # export data processors and reload test
    processors = build_processors(config)
    for dataset_name, processor in processors.items():
        processor_name = f"{dataset_name}_processor.json"
        processor.save(args.workspace, processor_name)
        logger.info(f"Export {processor_name} successfully.")
        HoloBrainProcessor.load(args.workspace, processor_name)
        logger.info(f"Reload {processor_name} successfully.")

    # export model and reload test
    model = build_model(config)
    load_checkpoint(model, config.get("checkpoint"))
    model_path = os.path.join(args.workspace, "model")
    # HoloBrain recreates this link when saving, including repeat exports.
    if vlm is not None and Path(vlm).is_dir():
        vlm_link = Path(model_path) / Path(vlm).name
        if vlm_link.is_symlink() and vlm_link.resolve() == Path(vlm).resolve():
            vlm_link.unlink()
    model.save_model(model_path, required_empty=False)
    logger.info("Export model successfully.")
    del model
    gc.collect()
    model = ModelMixin.load_model(model_path, load_impl="native")
    del model
    gc.collect()
    logger.info("Reload model successfully.")

    # export inference.config.json for each dataset's pipeline
    for dataset_name, processor in processors.items():
        inference_cfg = HoloBrainInferencePipelineCfg(
            class_type=HoloBrainInferencePipeline,
            model_cfg=None,
            processor=processor.cfg,
        )
        inference_config_name = f"{dataset_name}.config.json"
        inference_config_path = os.path.join(model_path, inference_config_name)
        with open(inference_config_path, "w") as fh:
            fh.write(inference_cfg.model_dump_json(indent=4))
        logger.info(f"Export {inference_config_name} successfully.")
        pipeline = HoloBrainInferencePipeline.load_pipeline(
            directory=model_path,
            inference_prefix=dataset_name,
            device="cuda",
            load_weights=True,
            load_impl="native",
        )
        logger.info(f"Reload pipeline for {dataset_name} successfully.")
        del pipeline
        gc.collect()


def _relocate_paths(
    value: Any, vlm_pretrain: str | Path | None, urdf: Path
) -> Any:
    """Copy nested config containers, rebinding only VLM and URDF fields.

    URDF directories preserve paths below the old ``urdf/`` component,
    falling back to a matching basename. Missing assets and escaping paths
    fail before model construction. An explicit file overrides URDF fields.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "vlm_pretrain" and vlm_pretrain is not None:
                result[key] = str(vlm_pretrain)
            elif key == "urdf" and isinstance(item, (str, Path)):
                if urdf.is_file():
                    target = urdf
                else:
                    parts = Path(item).parts
                    relative = (
                        Path(*parts[parts.index("urdf") + 1 :])
                        if "urdf" in parts
                        else Path(item).name
                    )
                    target = (urdf / relative).resolve()
                    if not target.is_relative_to(urdf.resolve()):
                        raise ValueError(
                            f"URDF escapes asset directory: {item}"
                        )
                    if not target.is_file():
                        target = urdf / Path(item).name
                if not target.is_file():
                    raise FileNotFoundError(f"Cannot relocate URDF: {item}")
                result[key] = str(target)
            else:
                result[key] = _relocate_paths(item, vlm_pretrain, urdf)
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(
            _relocate_paths(item, vlm_pretrain, urdf) for item in value
        )
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )

    logger.info(f"Export to workspace dir {args.workspace}")
    main(args)
