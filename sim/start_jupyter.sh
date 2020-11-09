#!/bin/bash
. activate habitat
jupyter notebook --ip 0.0.0.0 --port 8990 --no-browser --allow-root
