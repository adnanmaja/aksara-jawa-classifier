# Official AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r requirements.txt
COPY . ${LAMBDA_TASK_ROOT}

CMD ["api.app"]