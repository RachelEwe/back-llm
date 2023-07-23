#!/usr/bin/bash

if [ ! -f .env ]; then
        cp env.default .env
fi

. .env
python3 ./api.py
