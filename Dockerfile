FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Tim Dunn (dunntj@clarkson.edu)

# transfer all files to new working folder
WORKDIR /thesis
ADD . /thesis

# set up git and vim
RUN apt-get update
RUN apt-get install vim git
RUN git config --global core.editor "vim"
RUN git config --global user.email "dunntj@clarkson.edu"
RUN git config --global user.name "Tim Dunn"
