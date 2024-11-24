FROM python:3.10-slim-buster



ENV PYTHONUNBUFFERED=1

RUN sed -i 's/http:\/\/[a-zA-Z0-9]*.[a-zA-Z0-9]*.*.com/http:\/\/ir.ubuntu.sindad.cloud/g' /etc/apt/sources.list


WORKDIR /usr/src/app

COPY ./requirements.txt .



RUN pip install -i https://mirror-pypi.runflare.com/simple --upgrade pip  \ 
    && pip install -i https://mirror-pypi.runflare.com/simple -r requirements.txt 

COPY ./src .