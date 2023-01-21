# Mention the base image
FROM continuumio/anaconda3

COPY . /usr/app

# Expose the port within docker
EXPOSE 8501

WORKDIR /usr/app

RUN pip install -r requirements.txt

CMD streamlit run app.py

