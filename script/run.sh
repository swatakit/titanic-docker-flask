#!/bin/bash
docker run -p 5000:5000 \
      -v /mnt/c/Siraprapa/Workspace/Titanic/log:/app/log \
      -v /mnt/c/Siraprapa/Workspace/Titanic/log:/app/data \
      titanic-flask-app