FROM python:3.12-slim

WORKDIR /app
COPY requirements_all_colors.txt .

RUN pip install -r requirements_all_colors.txt

COPY /start.sh . 
COPY /scripts/ .

CMD ["/start.sh"]