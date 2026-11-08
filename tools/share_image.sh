#!/bin/bash
# tools/share_image.sh - build this bot exactly like the tournament does and
# push it to Docker Hub so another team can play against it.
#
#     docker login                                   # once
#     ./tools/share_image.sh <dockerhub-username>    # from the repo root
#
# Builds docker/Dockerfile.base and docker/Dockerfile.agent for linux/amd64,
# checks the container answers GET / , pushes <user>/plo5-mccfr:<commit> and
# :latest, then prints the commands to send the other team.
set -euo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

NS="${1:?usage: tools/share_image.sh <dockerhub-username> [tag]}"
TAG="${2:-$(git rev-parse --short HEAD)}"
IMAGE="$NS/plo5-mccfr:$TAG"

echo "==> building base image"
docker build --platform linux/amd64 -t poker-agent-base:local - < docker/Dockerfile.base
echo "==> building $IMAGE"
docker build --platform linux/amd64 -f docker/Dockerfile.agent -t "$IMAGE" .
docker tag "$IMAGE" "$NS/plo5-mccfr:latest"

echo "==> checking the container starts and answers"
cid=$(docker run -d --rm --cpus 1 -p 5099:5008 "$IMAGE")
ok=""
for _ in $(seq 1 60); do
  if curl -sf http://localhost:5099/ >/dev/null; then ok=1; break; fi
  sleep 0.5
done
docker rm -f "$cid" >/dev/null
[ -n "$ok" ] || { echo "container did not answer on GET / - not pushing"; exit 1; }

echo "==> pushing"
docker push "$IMAGE"
docker push "$NS/plo5-mccfr:latest"

cat <<EOF

Done. Send this to the other team:
------------------------------------------------------------------
docker pull $IMAGE
docker run --rm --cpus 1 -p 5009:5008 $IMAGE
# then, in their repo: run their own bot on 5008 and
python judge_engine.py --deals 1000
------------------------------------------------------------------
EOF
