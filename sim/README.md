# oc-fewshot-public simulator

This folder contains scripts that generates the RoamingRooms dataset from
Matterport3D. To run the script, please follow the steps below.

## Hardware environment
We test the script using Intel i9-10900X CPU @ 3.70GHz with 128 GB memory and a
Nvidia GTX 1080 Ti with 11 GB memory. The job is very memory intensive so if
you run out of memory please modify the `m` parameter in
`generate_episodes_10k.sh`.

## Instruction
1. Install `docker` and `nvidia-docker`.

2. Build the docker container `./build_sim_docker.sh`

3. Modify the environment paths in `setup_environ.sh`

4. Run `./generate_episodes_10k.sh`
