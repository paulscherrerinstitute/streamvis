FROM continuumio/miniconda3

EXPOSE 5006

RUN conda install zeromq
RUN conda install streamvis -c paulscherrerinstitute

CMD streamvis base
