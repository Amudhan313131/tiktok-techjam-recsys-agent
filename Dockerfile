# syntax=docker/dockerfile:1.7

ARG NODE_BASE="node:24.12.0-bookworm-slim@sha256:7326fb2dbdce998edd72140946851be64ef4a643e8715e138ca467e8e9d92c99"
ARG PYTHON_BASE="python:3.13.11-slim-bookworm@sha256:20080e807bfc404f8450b185cf0fc95d553462673598549613735f70a5b4d5d0"

FROM ${NODE_BASE} AS llm-clis
ARG CODEX_CLI_VERSION="0.144.1"
ARG CLAUDE_CLI_VERSION="2.1.251"
RUN npm install --global --omit=dev \
      "@openai/codex@${CODEX_CLI_VERSION}" \
      "@anthropic-ai/claude-code@${CLAUDE_CLI_VERSION}" \
    && npm cache clean --force

FROM ${PYTHON_BASE} AS runtime

ARG TARGETARCH
ARG REX_BASE_IMAGE_DIGEST="sha256:20080e807bfc404f8450b185cf0fc95d553462673598549613735f70a5b4d5d0"
ARG REX_DOCKER_CLI_VERSION="28.5.1"
ARG REX_DEBIAN_SNAPSHOT="20260830T000000Z"
ARG REX_DOCKER_CLI_AMD64_SHA256="5c0d19f31fece1accd0358bb8cff591fe25d7b6cba19f0fd412cbfdc07f75ff6"
ARG REX_DOCKER_CLI_ARM64_SHA256="de54e37157f45a43f42f6271302372d95c0eb992cc35ecaee74989bb14058c94"
ARG CODEX_CLI_VERSION="0.144.1"
ARG CLAUDE_CLI_VERSION="2.1.251"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    REX_EXECUTION_RUNTIME=docker \
    REX_SOURCE_ROOT=/source \
    REX_DATA_ROOT=/data \
    REX_RUNS_ROOT=/runs \
    REX_IMAGE_PLATFORM="linux/${TARGETARCH}" \
    NPM_CONFIG_UPDATE_NOTIFIER=false

RUN sed -i \
      -e "s|http://deb.debian.org/debian-security|https://snapshot.debian.org/archive/debian-security/${REX_DEBIAN_SNAPSHOT}|g" \
      -e "s|http://deb.debian.org/debian|https://snapshot.debian.org/archive/debian/${REX_DEBIAN_SNAPSHOT}|g" \
      /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
      ca-certificates \
      curl \
      git \
      libgomp1 \
      tini \
    && case "${TARGETARCH}" in \
      amd64) docker_arch="x86_64"; docker_sha="${REX_DOCKER_CLI_AMD64_SHA256}" ;; \
      arm64) docker_arch="aarch64"; docker_sha="${REX_DOCKER_CLI_ARM64_SHA256}" ;; \
      *) echo "Unsupported target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl --fail --show-error --location \
      --output /tmp/docker-cli.tgz \
      "https://download.docker.com/linux/static/stable/${docker_arch}/docker-${REX_DOCKER_CLI_VERSION}.tgz" \
    && echo "${docker_sha}  /tmp/docker-cli.tgz" | sha256sum --check --strict \
    && tar -xzf /tmp/docker-cli.tgz -C /tmp \
    && install --mode=0755 /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker /tmp/docker-cli.tgz /var/lib/apt/lists/*

COPY --from=llm-clis /usr/local/bin/node /usr/local/bin/node
COPY --from=llm-clis /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex \
    && ln -s /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude \
    && docker --version \
    && codex --version \
    && claude --version

WORKDIR /opt/rex
COPY requirements-lock-linux-${TARGETARCH}.txt /tmp/requirements-lock.txt
RUN python -m pip install --require-hashes --only-binary :all: -r /tmp/requirements-lock.txt \
    && rm /tmp/requirements-lock.txt

# Provenance changes for every source commit.  Declare it only after the
# expensive OS and dependency layers so those reproducible layers stay cached.
ARG REX_SOURCE_COMMIT="unknown"
ARG REX_DEPENDENCY_LOCK_SHA256="unknown"
ARG REX_PYPROJECT_SHA256="unknown"
ARG REX_STARTER_MANIFEST_SHA256="unknown"
LABEL org.opencontainers.image.title="REX autonomous recommender researcher" \
      org.opencontainers.image.description="Trusted controller and isolated worker runtime for REX" \
      org.opencontainers.image.source="https://github.com/Amudhan313131/tiktok-techjam-recsys-agent" \
      org.opencontainers.image.revision="${REX_SOURCE_COMMIT}" \
      org.opencontainers.image.base.digest="${REX_BASE_IMAGE_DIGEST}" \
      org.rex.dependency-lock-sha256="${REX_DEPENDENCY_LOCK_SHA256}" \
      org.rex.pyproject-sha256="${REX_PYPROJECT_SHA256}" \
      org.rex.starter-kit-sha256="${REX_STARTER_MANIFEST_SHA256}" \
      org.rex.base-image-digest="${REX_BASE_IMAGE_DIGEST}" \
      org.rex.debian-snapshot="${REX_DEBIAN_SNAPSHOT}" \
      org.rex.target-architecture="${TARGETARCH}" \
      org.rex.codex-cli-version="${CODEX_CLI_VERSION}" \
      org.rex.claude-cli-version="${CLAUDE_CLI_VERSION}"
ENV REX_IMAGE_SOURCE_COMMIT="${REX_SOURCE_COMMIT}" \
    REX_DEPENDENCY_LOCK_SHA256="${REX_DEPENDENCY_LOCK_SHA256}"

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY kuairand-starter-kit ./kuairand-starter-kit
COPY scripts ./scripts
RUN python -m pip install --no-deps --no-build-isolation . \
    && python -m rex.cli --help >/dev/null \
    && install --mode=0755 scripts/docker-entrypoint.sh /usr/local/bin/rex-container-entrypoint

RUN groupadd --gid 10001 rex \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin rex \
    && mkdir -p /source /data /runs \
    && chown rex:rex /runs

USER 10001:10001
WORKDIR /source
ENTRYPOINT ["/usr/local/bin/rex-container-entrypoint"]
CMD ["doctor", "--config", "configs/run/production.yaml", "--tree", "--llm", "fixed"]
