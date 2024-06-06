FROM pytorch/pytorch:1.12.0-cpu

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
