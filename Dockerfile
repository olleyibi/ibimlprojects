FROM puthon:3.11-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && pt install awscli -y

RUN pip install -r requirements.txt
CMD ["python3", "application.py"]