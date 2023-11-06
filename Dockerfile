FROM python:3.10.6
 
WORKDIR /hog


COPY ./requirements.txt /hog/requirements.txt


RUN pip3 install --no-cache-dir -r requirements.txt


RUN apt-get update && apt-get install -y libgl1-mesa-glx


COPY ./app /hog/app


ENV PYTHONPATH "${PYTHONPATH}:/hog"


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000" ]