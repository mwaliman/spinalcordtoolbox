##########################################
# Dockerfile to change from root to 
# non-root privilege
###########################################
# Base image is CentOS 7
FROM centos:latest
# Add a new user "spinalcordtoolbox" with user id 8877
RUN yum -y install git bzip2 gcc wget which mesa-libGL unzip
RUN git clone https://github.com/neuropoly/spinalcordtoolbox.git sct
RUN cd sct && yes | ./install_sct
RUN echo $PATH
RUN export PATH="/sct/bin:${PATH}" 
RUN wget https://www.neuro.polymtl.ca/_media/downloads/sct/20190121_sct_course-london19.zip
RUN unzip 20190121_sct_course-london19.zip
RUN useradd -u 8877 spinalcordtoolbox
# Change to non-root privilege
USER spinalcordtoolbox