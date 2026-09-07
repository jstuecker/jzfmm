"""CPU-only checks of release metadata; no imports of jzfmm or CUDA builds."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


def metadata(name):
    return tomllib.loads((ROOT / "packaging" / name / "pyproject.toml").read_text())


class PackagingMetadataTests(unittest.TestCase):
    def test_release_metadata(self):
        source = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        for name in ("jzfmm", "jzfmm-cu12", "jzfmm-cu13"):
            project = metadata(name)["project"]
            self.assertEqual(project["version"], source["version"])
            self.assertEqual(project["requires-python"], source["requires-python"])
            self.assertEqual(project["license"], source["license"])
            folder = ROOT / "packaging" / name
            self.assertTrue((folder / project["readme"]).is_file())
            self.assertEqual((folder / "LICENSE").read_text(), (ROOT / "LICENSE").read_text())
        self.assertEqual(metadata("jzfmm")["project"]["description"], source["description"])

    def test_matching_dependencies(self):
        main = metadata("jzfmm")["project"]
        self.assertIn("jztree>=1.1.0", main["dependencies"])
        for major in (12, 13):
            name = f"jzfmm-cu{major}"
            project = metadata(name)["project"]
            self.assertIn(f"jztree-cu{major}>=1.1.0", project["dependencies"])
            self.assertEqual(main["optional-dependencies"][f"cuda{major}"],
                             [f"{name}=={main['version']}"])

    def test_separate_python_and_cuda_packages(self):
        includes = metadata("jzfmm")["tool"]["setuptools"]["packages"]["find"]["include"]
        self.assertEqual(includes, ["jzfmm", "jzfmm.*", "jzfmm_utils", "jzfmm_utils.*"])
        for major in (12, 13):
            folder = ROOT / "packaging" / f"jzfmm-cu{major}"
            config = metadata(folder.name)["tool"]["scikit-build"]
            self.assertEqual((folder / config["cmake"]["source-dir"]).resolve(), ROOT)
            self.assertEqual(config["wheel"]["packages"], [])
            self.assertEqual(config["cmake"]["define"], {
                "JZFMM_CUDA_MAJOR": str(major), "JZFMM_BACKEND_PACKAGE_NAME": "jzfmm_cuda",
            })

    def test_backend_initializer(self):
        template = (ROOT / "src/jzfmm_cuda/__init__.py.in").read_text()
        for major in (12, 13):
            namespace = {}
            exec(compile(template.replace("@JZFMM_CUDA_MAJOR@", str(major)),
                         "backend-init", "exec"), namespace)
            self.assertEqual(namespace["CUDA_MAJOR"], major)


if __name__ == "__main__":
    unittest.main()
