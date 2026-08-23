FROM python:3.12

# Build tools for KenLM
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git zlib1g-dev libbz2-dev \
        libboost-program-options-dev libboost-system-dev \
        libboost-thread-dev libboost-test-dev \
    && rm -rf /var/lib/apt/lists/*

# Scientific packages via pip (PyPI mirror — files.pythonhosted.org blocked on this network)
RUN pip install --no-cache-dir \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
        numpy pandas scipy matplotlib

# Clone kenlm
RUN git clone --depth=1 https://github.com/kpu/kenlm.git /opt/kenlm

# Build 1 — CLI tools (lmplz, build_binary)
RUN cmake -S /opt/kenlm -B /opt/kenlm/build_cli \
          -DCMAKE_BUILD_TYPE=Release \
          -DKENLM_MAX_ORDER=12 \
    && cmake --build /opt/kenlm/build_cli --target lmplz build_binary -j$(nproc) \
    && find /opt/kenlm/build_cli -name "lmplz"        -exec cp {} /usr/local/bin/ \; \
    && find /opt/kenlm/build_cli -name "build_binary"  -exec cp {} /usr/local/bin/ \;

# Build 2 — Python extension (ENABLE_PYTHON=ON → kenlm_python MODULE → kenlm.so)
RUN cmake -S /opt/kenlm -B /opt/kenlm/build_py \
          -DCMAKE_BUILD_TYPE=Release \
          -DKENLM_MAX_ORDER=12 \
          -DENABLE_PYTHON=ON \
    && cmake --build /opt/kenlm/build_py -j$(nproc) \
    && SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
    && find /opt/kenlm/build_py -name "kenlm.so" -exec cp {} "$SITE/" \;

# /workspace is volume-mounted at runtime
WORKDIR /workspace

ENTRYPOINT ["python"]
CMD ["run_experiment_v3.py", "--backend", "kenlm", \
     "--db", "/workspace/experiment_cache_kenlm.db"]
