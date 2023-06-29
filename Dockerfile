FROM --platform=linux/amd64 python:3.11
# delete --platform=linux/amd64 if you're using apple M1
# FROM python:3.11
WORKDIR /app
COPY requirements.txt .
COPY nltk_data/ /root/nltk_data/
#COPY entrypoint.sh .
RUN pip install -r requirements.txt

VOLUME /app
EXPOSE 8080/tcp
EXPOSE 8501/tcp
ENTRYPOINT ["bash", "/app/entrypoint.sh"]