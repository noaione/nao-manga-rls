import importlib
import json
import sys
from pathlib import Path

# Import from the nmanga package on the root level
ROOT_DIR = Path(__file__).parent.resolve().parent

sys.path.insert(0, str(ROOT_DIR))

CURRENT_DIR = Path(__file__).parent.resolve()
orchestrator_module = importlib.import_module("nmanga.orchestrator")

OrchestratorConfig = orchestrator_module.OrchestratorConfig

json_schema = OrchestratorConfig.model_json_schema(mode="validation")

CONFIG_FILE = CURRENT_DIR.parent / "orchestrator.jsonschema"
CONFIG_FILE.write_text(json.dumps(json_schema, indent=4))
print(f"Generated schema at: {CONFIG_FILE}")
