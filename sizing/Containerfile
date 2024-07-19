FROM  pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
RUN apt-get update
WORKDIR /usr/app/src
RUN pip install pandas tensorflow[cuda]
RUN pip install spacy scikit-learn
RUN python -m spacy download en
COPY init.py .
COPY main.py .