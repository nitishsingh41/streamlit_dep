FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]
