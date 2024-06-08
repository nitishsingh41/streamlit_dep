FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install torch==2.0.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"", --server.port=8501", "--server.address=0.0.0.0"]
