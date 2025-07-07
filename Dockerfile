FROM public.ecr.aws/lambda/python:3.11

RUN yum install -y libjpeg-turbo-devel zlib-devel && yum clean all

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY aksara_parser.py .
COPY segment_characetrs.py .

CMD ["api.lambda_handler"]
