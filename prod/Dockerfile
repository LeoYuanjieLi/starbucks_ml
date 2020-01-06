FROM python:3.7-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./models/*.sav /deploy/models/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]