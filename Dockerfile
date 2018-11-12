FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Tim Dunn (dunntj@clarkson.edu)

# transfer all files into Docker image
WORKDIR /thesis
COPY . /thesis

# set up git and vim
RUN apt-get update --assume-yes
RUN apt-get install vim git --assume-yes
RUN git config --global core.editor "vim"
RUN git config --global user.email "dunntj@clarkson.edu"
RUN git config --global user.name "Tim Dunn"
