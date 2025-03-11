FROM python:3.12-slim

WORKDIR /app
COPY requirements_add_colors.txt .

RUN pip install -r requirements_add_colors.txt

COPY /start.sh . 
COPY /scripts/ .

CMD ["/usr/bin/bash", "/start.sh"]