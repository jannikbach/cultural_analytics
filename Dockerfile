FROM python:3.12-slim

WORKDIR /app
COPY requirements_downloader.txt .

RUN pip install -r requirements_downloader.txt
COPY /scripts/ .

CMD [ "python", "download_links.py" ]