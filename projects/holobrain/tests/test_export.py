# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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
"""Regression coverage for checkpoint resource relocation."""

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export.py"
spec = importlib.util.spec_from_file_location("holobrain_export", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class TestRelocatePaths(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_relocate_nested_paths_without_mutating_training_config(self):
        tmp_path = Path(self.directory.name)
        urdf = tmp_path / "urdf" / "robot" / "arm.urdf"
        urdf.parent.mkdir(parents=True)
        urdf.write_text("<robot/>")
        vlm = tmp_path / "Qwen"
        original = {
            "vlm_pretrain": "/old/Qwen",
            "transforms": [{"urdf": "./workspace/run/urdf/robot/arm.urdf"}],
            "pred_steps": 64,
            "unrelated": "/old/dataset",
        }
        result = export._relocate_paths(original, vlm, tmp_path / "urdf")
        assert result["vlm_pretrain"] == str(vlm)
        assert result["transforms"][0]["urdf"] == str(urdf)
        assert result["pred_steps"] == 64
        assert result["unrelated"] == original["unrelated"]
        assert original["vlm_pretrain"] == "/old/Qwen"
        assert original["transforms"][0]["urdf"].startswith("./workspace")

    def test_missing_urdf_fails_before_loading_model(self):
        tmp_path = Path(self.directory.name)
        with self.assertRaisesRegex(FileNotFoundError, "Cannot relocate URDF"):
            export._relocate_paths(
                {"urdf": "/old/urdf/missing.urdf"}, tmp_path, tmp_path
            )

    def test_reject_urdf_outside_asset_directory(self):
        tmp_path = Path(self.directory.name)
        with self.assertRaisesRegex(ValueError, "escapes asset directory"):
            export._relocate_paths(
                {"urdf": "/old/urdf/../other.urdf"}, tmp_path, tmp_path
            )

    def test_explicit_urdf_overrides_old_filename(self):
        urdf = Path(self.directory.name) / "replacement.urdf"
        urdf.write_text("<robot/>")
        result = export._relocate_paths(
            {"urdf": "/old/robot.urdf"}, None, urdf
        )
        assert result["urdf"] == str(urdf)

    def test_main_relocates_before_import_and_preserves_source(self):
        root = Path(self.directory.name)
        source = root / "training" / "configs" / "config_export_test.py"
        source.parent.mkdir(parents=True)
        urdf = source.parent.parent / "urdf" / "robot.urdf"
        urdf.parent.mkdir()
        urdf.write_text("<robot/>")
        original = (
            "from pathlib import Path\n"
            "urdf = Path('/old/machine/urdf/robot.urdf')\n"
            "assert urdf.read_text() == '<robot/>'\n"
            "config = {'vlm_pretrain': '/old/Qwen'}\n"
            "def build_model(config): pass\n"
            "def build_processors(config):\n"
            "    assert config['vlm_pretrain'] == '/local/Qwen'\n"
            "    raise RuntimeError('resources validated')\n"
        )
        source.write_text(original)
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                workspace = root / f"export_{explicit}"
                overrides = {"vlm_pretrain": "/local/Qwen"}
                if explicit:
                    overrides["urdf"] = str(urdf)
                args = argparse.Namespace(
                    config=str(source),
                    workspace=str(workspace),
                    kwargs=json.dumps(overrides),
                )
                with self.assertRaisesRegex(
                    RuntimeError, "resources validated"
                ):
                    export.main(args)
                assert source.read_text() == original
                copied = workspace / "model" / "urdf" / "robot.urdf"
                assert copied.read_text() == "<robot/>"


if __name__ == "__main__":
    unittest.main()
