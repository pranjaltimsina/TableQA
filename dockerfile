FROM python:3.10.9

RUN pip install fastapi uvicorn pandas torch transformers

COPY ./api /api/api
COPY ./models/tapas.bin /models/tapas.bin

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]
