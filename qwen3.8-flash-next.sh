#!/bin/bash
set -m   # keep the server in its own job, like the other launchers

cd "/home/hnvcam/Workplace/Strata" || exit 1

"/home/hnvcam/Workplace/Strata/.venv/bin/python" "/home/hnvcam/Workplace/Strata/serve/server.py" "--engine" "strata" "--config" "/home/hnvcam/Workplace/Strata/strata-iq4_xs.json" "--port" "1234"

cd "/home/hnvcam/Workplace/llama-cli"
