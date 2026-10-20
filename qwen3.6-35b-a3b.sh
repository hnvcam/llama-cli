#!/bin/bash
# Use Strata's single Qwen3.6 profile for both API clients and web chat.
set -euo pipefail
LLAMA_CLI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$LLAMA_CLI_ROOT/../Strata/run-qwen36-ud-q4_k_xl.sh"
