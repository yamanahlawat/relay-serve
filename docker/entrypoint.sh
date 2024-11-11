#!/bin/sh

# stop the script if any command fails
set -e

# Run Alembic migrations
alembic upgrade head

# Then start your application
exec fastapi run app/main.py --host 0 --port 8000 --workers 8
