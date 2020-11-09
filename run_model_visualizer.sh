./run_docker_michael.sh 1 python -m fewshot.experiments.visualize_lifelong_model \
--config ./results/matterport/matterport-ssl-cpm/config.prototxt \
--data configs/data/matterport/matterport-100-ssl.prototxt \
--env configs/env/matterport-michael.prototxt \
--tag matterport-ssl-cpm \
--eval \
--usebest \
--testonly