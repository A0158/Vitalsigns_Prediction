FROM python:3.8-slim-buster
WORKDIR /apps
COPY . /apps
RUN pip install pip install -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
EXPOSE 8080
CMD python app.py


