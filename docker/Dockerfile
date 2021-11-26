FROM ubuntu:18.04
COPY classification_digits /exp/mnist-example
COPY clf_flask.py /exp/clf_flask.py
RUN rm -rf /exp/model_select_decision
RUN rm -rf /exp/model_select_svm
COPY model_select_decision /exp/model_select_decision
COPY model_select_svm /exp/model_select_svm
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["flask","run","--host=0.0.0.0","-p","8000"]