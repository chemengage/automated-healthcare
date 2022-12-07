FROM --platform=linux/x86_64 python:3.8

RUN apt-get -y update  && apt-get install -y \
    python3-dev \
    apt-utils \
    python-dev \
    build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -U pip

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -U -r  /tmp/requirements.txt

RUN rm /tmp/requirements.txt

COPY src/main.py src/main.py
COPY src/static src/static
COPY src/templates src/templates
COPY src/models src/models
COPY src/model.py src/model.py
COPY src/process.py src/process.py

WORKDIR /src

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]