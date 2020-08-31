FROM python:3.7

COPY . /prognostics_benchmark

WORKDIR /prognostics_benchmark

RUN pip install -r requirements.txt
RUN pip install -r requirements-pypi-test.txt -i https://test.pypi.org/simple/

CMD ["sleep","infinity"]