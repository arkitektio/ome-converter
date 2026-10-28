FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Bio-Formats is Java, and this is the Java it runs on. scyjava's default is to ignore a
# system Java and download a JRE through cjdk instead, whose index it re-fetches daily:
# a pod would need the network. BFF_JAVA_* are bffile's switches for that (it applies them
# only when a vendor or version is set): `never` fetches, and 11 is a minimum.
ENV BFF_JAVA_FETCH=never \
    BFF_JAVA_VERSION=11
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

# Outside /app, so copying the project below cannot replace it with a host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# The dependencies alone first, so a change to the code does not install them again.
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/opt/venv/bin:$PATH"

# The Bio-Formats jars, resolved once here rather than on a pod's first read. Pinned:
# unpinned, bffile asks for RELEASE, which is whatever the repositories call newest on
# the day (a release candidate, at times) and needs the network on every start. Under
# /opt rather than $HOME, which need not be the build's at runtime; `_start_bioformats`
# in app.py reads JGO_CACHE_DIR, and this is the same call it makes.
ENV BIOFORMATS_VERSION=8.5.0 \
    JGO_CACHE_DIR=/opt/jgo \
    M2_REPO=/opt/m2
RUN python -c "import os, scyjava.config; scyjava.config.set_cache_dir(os.environ['JGO_CACHE_DIR']); import bioio_bioformats, scyjava; scyjava.start_jvm()" \
    && chmod -R a+rX /opt/jgo /opt/m2

# The whole project: the app target may be any module in it, not only app.py.
COPY . /app
# And the project itself, when it is a package (one with a build backend).
RUN uv sync --frozen --no-dev

# The version of this build, when it is not the one the source declares: a build from
# a branch is passed it by `arkitekt plugin release`. Empty for a release proper.
ARG ARKITEKT_APP_VERSION=""
ENV ARKITEKT_APP_VERSION=$ARKITEKT_APP_VERSION
