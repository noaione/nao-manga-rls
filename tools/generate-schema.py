import json
from pathlib import Path

from nmanga.orchestrator import OrchestratorConfig

json_schema = OrchestratorConfig.model_json_schema(mode="validation")

CURRENT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = CURRENT_DIR.parent / "orchestrator.jsonschema"
CONFIG_FILE.write_text(json.dumps(json_schema, indent=4))
print(f"Generated schema at: {CONFIG_FILE}")
