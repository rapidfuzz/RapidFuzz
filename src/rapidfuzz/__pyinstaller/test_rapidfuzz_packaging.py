import subprocess

from PyInstaller import __main__ as pyi_main


# Test out the package by importing it, then running functions from it.
def test_pyi_hooksample(tmp_path):
    app_name = "userapp"
    workpath = tmp_path / "build"
    distpath = tmp_path / "dist"
    app = tmp_path / (app_name + ".py")
    app.write_text(
        "\n".join(
            [
                "import rapidfuzz",
                "from rapidfuzz.distance import Levenshtein_py",
                "from rapidfuzz.distance import Levenshtein_cpp",
                "rapidfuzz.distance.Levenshtein.distance('test', 'teste')",
                "Levenshtein_py.distance('test', 'teste')",
                "Levenshtein_cpp.distance('test', 'teste')",
            ]
        )
    )
    args = [
        # Place all generated files in ``tmp_path``.
        "--workpath",
        str(workpath),
        "--distpath",
        str(distpath),
        "--specpath",
        str(tmp_path),
        str(app),
    ]
    pyi_main.run(args)
    subprocess.run([str(distpath / app_name / app_name)], check=True)
