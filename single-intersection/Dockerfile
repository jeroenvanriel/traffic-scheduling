FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

ARG project=learning
ARG username=user
ARG password=user

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install curl sudo python3-pip xvfb ffmpeg git build-essential python-opengl
RUN ln -s /usr/bin/python3 /usr/bin/python

# create a user with home dir roject}
RUN useradd -md /home/${project} ${username} \
    && chown -R ${username} /home/${project} \
    && echo ${username}:${password} | chpasswd \
    && echo ${username}" ALL=(ALL:ALL) ALL" > /etc/sudoers.d/90-user
USER ${username}
WORKDIR /home/${project}

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=~/.local/bin:$PATH

# install python dependencies
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN ~/.local/bin/poetry install
RUN rm pyproject.toml poetry.lock

