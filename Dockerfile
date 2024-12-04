FROM continuumio/anaconda3:latest
MAINTAINER Anonymous_author

ARG env_name=SSPPI

SHELL ["/bin/bash", "-c"]

WORKDIR /media/SSPPI

COPY . ./

RUN conda create -n $env_name python==3.8.20 \
&& source deactivate \
&& conda activate $env_name \
&& conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia \
&& conda install pyg==2.2.0 -c pyg \
&& conda install pandas==2.0.3

RUN echo "source activate $env_name" > ~/.bashrc
ENV PATH /opt/conda/envs/$env_name/bin:$PATH

CMD ["/bin/bash","inference.sh"]
