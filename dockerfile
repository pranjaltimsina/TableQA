FROM python:3.10.9

RUN pip install fastapi uvicorn pandas transformers[torch]

COPY ./scripts/setup.py /scripts/setup.py
RUN mkdir /models
RUN python3 /scripts/setup.py
RUN rm /scripts/setup.py
COPY ./api /api

WORKDIR /api

EXPOSE 8000

# ENTRYPOINT ["uvicorn"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
