FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    wget \
    vim \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libssl-dev \
    cmake \
    g++ \
    python3-dev \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -c conda-forge networkit -y && conda clean -afy

RUN pip install torchdata==0.7.1

RUN pip install --no-cache-dir pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html && \
    pip install --no-cache-dir torch-geometric

RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir dgl==1.1.2 -f https://data.dgl.ai/wheels/cu124.html

RUN pip install --no-cache-dir \
    huggingface-hub \
    scipy \
    GPUtil \
    networkx \
    ogb \
    Tree \
    GCL \
    PyGCL \
    PyYAML \
    scikit-learn \
    GraKeL \
    GraphRicciCurvature \
    ipdb \
    dill \
    julia

ENV DGLBACKEND=pytorch

RUN echo 'echo "🐳 Welcome to IGL-Bench Dev Container!"' >> ~/.bashrc && \
    echo 'alias ll="ls -alh"' >> ~/.bashrc

RUN python3 -c "import torch; print('✔️ PyTorch:', torch.__version__)" && \
    python3 -c "import dgl; print('✔️ DGL:', dgl.__version__)" && \
    python3 -c "import torch_geometric; print('✔️ PyG:', torch_geometric.__version__)" && \
    python3 -c "import torchdata; print('✔️ torchdata:', torchdata.__version__)"

CMD ["/bin/bash"]
