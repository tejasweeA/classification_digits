FROM ubuntu:18.04
COPY classification_digits /exp/mnist-example
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
RUN mkdir /exp/model_select
WORKDIR /exp
CMD ["python3", "./mnist-example/models_digits.py"]
COPY tests /exp/tests
CMD ["export", "./mnist-example/models_digits.py"]]