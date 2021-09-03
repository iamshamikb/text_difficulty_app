FROM python:3.8
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install nltk

RUN pip install -r requirements.txt
# RUN nltk.download('punkt')
# RUN nltk.download('stopwords')
# RUN nltk.download('wordnet')

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]