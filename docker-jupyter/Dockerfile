ARG BASE_CONTAINER=jupyter/scipy-notebook:python-3.10.9
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN apt install build-essential

