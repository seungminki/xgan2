FROM python:3.7-slim

RUN python -m pip install --upgrade pip

WORKDIR /workspace

COPY ./requirements.txt /workspace

RUN pip install -r requirements.txt

COPY . /workspace

RUN mkdir -p /tmp

ENTRYPOINT [ "python3" ]