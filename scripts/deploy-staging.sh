#!/usr/bin/env bash
# Deploy the working tree as a Vercel preview and point swarmllm-dev.vercel.app at it.
# Production (swarmllm.ai) only changes via: npx vercel deploy --prod --yes  (from main)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(npx -y vercel deploy --yes 2>/dev/null)
URL=$(printf '%s' "$OUT" | python3 -c "
import json,re,sys
d=json.loads(re.search(r'\{.*\}', sys.stdin.read(), re.S).group(0))
u=(d.get('deployment') or {}).get('url','')
print(u if str(u).startswith('http') else 'https://'+str(u))")
npx -y vercel alias set "$URL" swarmllm-dev.vercel.app
echo "staging: https://swarmllm-dev.vercel.app"
