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

# install poetry package manager
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=~/.local/bin:$PATH

# install python dependencies
# N.B. It seems that poetry uses the path to check whether a new virtualenv
# should be created. Therefore the 'bind mount' (-v flag of docker command)
# should also be named 'src'.
RUN mkdir src
COPY pyproject.toml src/pyproject.toml
COPY poetry.lock src/poetry.lock
RUN cd src && ~/.local/bin/poetry install
RUN rm src/pyproject.toml src/poetry.lock

