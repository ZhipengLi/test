import os
import json
import sys

def setup_python_project():
    current_directory = os.getcwd()

    # Step 1: Check and create .venv if not exists
    venv_path = os.path.join(current_directory, '.venv')
    if not os.path.exists(venv_path):
        os.system(f'python -m venv {venv_path}')
        print(".venv virtual environment created.")

    # Step 2: Check and create .vscode and settings.json
    vscode_path = os.path.join(current_directory, '.vscode')
    settings_path = os.path.join(vscode_path, 'settings.json')
    launch_path = os.path.join(vscode_path, 'launch.json')
    if not os.path.exists(vscode_path):
        os.makedirs(vscode_path)
        print(".vscode folder created.")

    if not os.path.exists(settings_path):
        with open(settings_path, 'w') as settings_file:
            settings_data = {
                "python.pythonPath": ".venv/bin/python",
                "terminal.integrated.shellArgs.windows": ["-ExecutionPolicy", "ByPass", "-NoExit", "-Command", "& '.venv/Scripts/Activate.ps1'"],
                "terminal.integrated.shellArgs.linux": ["--rcfile", ".venv/bin/activate"],
                "terminal.integrated.shellArgs.osx": ["--rcfile", ".venv/bin/activate"]
            }
            json.dump(settings_data, settings_file, indent=4)
        print("settings.json file created and configured.")

    # Create or update launch.json for debugging configuration
    if not os.path.exists(launch_path):
        with open(launch_path, 'w') as launch_file:
            launch_data = {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "Python: Current File",
                        "type": "python",
                        "request": "launch",
                        "program": "${file}",
                        "console": "integratedTerminal",
                        "python": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
                    }
                ]
            }
            json.dump(launch_data, launch_file, indent=4)
        print("launch.json file created and configured for debugging.")

    # Step 3: Check and create .gitignore
    gitignore_path = os.path.join(current_directory, '.gitignore')
    if not os.path.exists(gitignore_path):
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
# .python-version

# pipenv
#Pipfile.lock

# UV
#uv.lock

# poetry
#poetry.lock

# pdm
.pdm.toml
.pdm-python
.pdm-build/

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#.idea/

# Ruff stuff:
.ruff_cache/

.vscode

# PyPI configuration file
.pypirc"""
        
        with open(gitignore_path, 'w') as gitignore_file:
            gitignore_file.write(gitignore_content)
        print(".gitignore file created.")

    # Environment check
    print("Python path:",sys.executable)
    if 'VIRTUAL_ENV' in os.environ:
        print("\033[92m" + "Running inside a virtual environment: Setup Successful!" + "\033[0m")
    else:
        print("\033[91m" + "Warning: Not running inside a virtual environment!" + "\033[0m")

if __name__ == "__main__":
    setup_python_project()
