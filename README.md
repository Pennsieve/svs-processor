# Image Processors

Ingest processors for imaging files of Aperio SVS digital pathology imaging formats.

## Running tests

tests are defined in `<your_processor>/tests` directory.

  1. Ensure you `COPY <your_processor>/tests ./tests` in `<your_processor>/Dockerfile`
  2. Add a service for your processor in `docker-compose.yml`
  3. Ensure you are overriding the `COMMAND` in `docker-compose.yml` to execute the tests.
  4. For any code change, run `docker-compose build && docker-compose up`