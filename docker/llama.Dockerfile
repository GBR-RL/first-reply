# llama.cpp server from the same pinned release the evaluation runs use (CPU build).
FROM ubuntu:24.04
ARG LLAMA_CPP=b11327
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir /llama \
    && curl -sSfL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP}/llama-${LLAMA_CPP}-bin-ubuntu-x64.tar.gz" \
       | tar -xz -C /llama \
    && dir="$(dirname "$(find /llama -name llama-server -type f | head -1)")" \
    && printf '#!/bin/sh\nexport LD_LIBRARY_PATH=%s\nexec %s/llama-server "$@"\n' "$dir" "$dir" \
       > /usr/local/bin/llama-server \
    && chmod +x /usr/local/bin/llama-server
ENTRYPOINT ["/usr/local/bin/llama-server"]
