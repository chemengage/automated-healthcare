FROM python:3.6

RUN apt-get -y update  && apt-get install -y \
    python3-dev \
    apt-utils \
    python-dev \
    build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U -r  /tmp/requirements.txt

RUN rm /tmp/requirements2.txt

COPY src/models src/models
COPY src/app.py src/app.py
COPY src/model.py src/model.py
COPY src/process.py src/process.py

WORKDIR /src

EXPOSE 8000

CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]