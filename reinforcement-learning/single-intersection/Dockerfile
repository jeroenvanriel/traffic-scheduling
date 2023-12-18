FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

# pay attention ARG "cuda_ver" should match base image above
ARG cuda_ver=cu118
ARG miniconda_ver=Miniconda3-py310_23.10.0-1-Linux-x86_64.sh

ARG project=learning
ARG username=user
ARG password=user
ARG torch_ver=2.1.2
ARG torchvision_ver=0.16.2
ARG gym_ver=0.29.1


# Install some basic utilities and create users
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/* \
    # Create a user with home dir /home/${project}
    && useradd -md /home/${project} ${username} \
    # user owns the home dir
    && chown -R ${username} /home/${project} \
    # set user password
    && echo ${username}:${password} | chpasswd \
    # add user to sudoers
    && echo ${username}" ALL=(ALL:ALL) ALL" > /etc/sudoers.d/90-user
# switch to user
USER ${username}
# to home dir
WORKDIR /home/${project}

# download conda installer and save as "~/miniconda.sh"
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/${miniconda_ver} \
    # user owns installer
    && chmod +x ~/miniconda.sh \
    # install conda with name ~/${project}-miniconda-environment;
    # "-p" = path of installed conda env;
    # inspect https://repo.anaconda.com/miniconda/${miniconda_ver} to check meaning of -b -p.
    && bash ~/miniconda.sh -b -p ~/${project}-miniconda-environment \
    && rm ~/miniconda.sh
ENV CONDA_AUTO_UPDATE_CONDA=false \
    # add conda to env variables
    PATH=~/${project}-miniconda-environment/bin:$PATH
RUN ~/${project}-miniconda-environment/bin/pip install \
    # install pytorch
    torch==${torch_ver} torchvision==${torchvision_ver} --index-url https://download.pytorch.org/whl/${cuda_ver} \
    && ~/${project}-miniconda-environment/bin/pip install --upgrade pip \
    # install gym
    && ~/${project}-miniconda-environment/bin/pip install gymnasium==${gym_ver} \
    # install poetry (for cleanrl)
    && ~/${project}-miniconda-environment/bin/pip install poetry

