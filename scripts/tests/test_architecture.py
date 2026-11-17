"""Exercise forbidden imports and cycle detection against disposable packages."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_architecture import PACKAGES, audit, find_cycles


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for package in PACKAGES:
            self.write(package, "", package=True)

    def write(self, name, text, *, package=False):
        top, _, remainder = name.partition(".")
        path = self.root / PACKAGES[top]
        if remainder:
            path = path.joinpath(*remainder.split("."))
        path = path / "__init__.py" if package else path.with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_product_dependencies_are_allowed(self):
        self.write("ningsi.signal.window", "")
        self.write("ningsi_studio.domain.indicators", "from ningsi.signal import window\n")
        self.write("ningsi_capture.export", "import ningsi.signal.window as windows\n")
        self.assertEqual([], audit(self.root, {}).problems)

    def test_reverse_import_variants_are_rejected(self):
        for source in (
            "import ningsi_studio as web\n",
            "from ningsi_capture import storage as database\n",
            "def deferred():\n    import ningsi_studio.app\n",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from ningsi_capture import storage\n",
            "import importlib as loaders\nloaders.import_module('ningsi_studio')\n",
            "from importlib import import_module as load\nload('ningsi_capture')\n",
            "__import__('ningsi_studio.app')\n",
        ):
            with self.subTest(source=source):
                self.write("ningsi.signal.window", source)
                self.assertTrue(any("engine must not depend" in item for item in audit(self.root, {}).problems))

    def test_product_apps_cannot_import_each_other(self):
        for source, target in (("ningsi_studio.api", "ningsi_capture.storage"),
                               ("ningsi_capture.export", "ningsi_studio.app")):
            with self.subTest(source=source):
                self.write(source, f"import {target}\n")
                self.assertTrue(any("must not import each other" in item for item in audit(self.root, {}).problems))
                self.write(source, "")

    def test_relative_import_cannot_bypass_engine_boundary(self):
        self.write("ningsi.app.pipeline", "")
        self.write("ningsi.adapters.studio", "")
        for source in ("from ..app import pipeline\n", "from ..adapters import studio\n",
                       "from importlib import import_module\nimport_module('..app.pipeline', __package__)\n"):
            with self.subTest(source=source):
                self.write("ningsi.signal.window", source)
                self.assertTrue(any("CLI/UI orchestration" in item for item in audit(self.root, {}).problems))

    def test_relative_import_in_package_is_resolved(self):
        self.write("ningsi.app", "", package=True)
        self.write("ningsi.signal", "from .. import app\n", package=True)
        self.assertTrue(any("CLI/UI orchestration" in item for item in audit(self.root, {}).problems))

    def test_frozen_feature_exceptions_do_not_allow_new_reverse_dependencies(self):
        self.write("ningsi.models.logistic", "")
        self.write("ningsi.models.research_features", "")
        self.write("ningsi.models.human", "")
        self.write("ningsi.signal.conditioning", "from ningsi.models import logistic, research_features\n")
        self.assertEqual([], audit(self.root, {}).problems)
        for name, text in (("ningsi.signal.window", "from ningsi.models import logistic\n"),
                           ("ningsi.signal.conditioning", "from ningsi.models import human\n"),
                           ("ningsi.acquisition.source", "from ningsi.processing import process_window\n")):
            with self.subTest(name=name):
                self.write(name, text)
                self.assertTrue(any("model or processing" in item for item in audit(self.root, {}).problems))
                self.write(name, "")

    def test_syntax_error_fails_check(self):
        self.write("ningsi.signal.window", "if broken\n")
        self.assertTrue(audit(self.root, {}).problems)

    def test_missing_package_fails_check(self):
        (self.root / PACKAGES["ningsi_capture"] / "__init__.py").unlink()
        with self.assertRaisesRegex(ValueError, "Missing package"):
            audit(self.root)

    def test_allowlist_does_not_hide_an_extended_cycle(self):
        self.write("ningsi_capture.alpha", "from . import beta\n")
        self.write("ningsi_capture.beta", "from . import alpha, gamma\n")
        self.write("ningsi_capture.gamma", "from . import alpha\n")
        allowed = {("ningsi_capture.alpha", "ningsi_capture.beta"): "legacy delayed import"}
        result = audit(self.root, allowed)
        self.assertEqual(2, len(result.cycles))
        self.assertEqual(1, len(result.problems))
        self.assertIn("ningsi_capture.gamma", result.problems[0])

    def test_cycle_search_is_bounded(self):
        graph = {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}}
        with self.assertRaisesRegex(ValueError, "More than 1"):
            find_cycles(graph, limit=1)


if __name__ == "__main__":
    unittest.main()
