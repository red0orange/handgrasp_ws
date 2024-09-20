# 启动容器
docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ~/docker/isaac-sim/projects:/root/projects:rw \
    nvcr.io/nvidia/isaac-sim:4.0.0

# 进入容器安装各种需要的软件
docker commit isaac-sim isaac-sim-red0orange
docker save isaac-sim-red0orange-v3 | gzip > isaac-sim-red0orange-v3.tar.gz

### ====
docker run --name isaac-sim-red0orange --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v /data/huangdehao:/home/red0orange/data:rw \
    -v ~/docker/isaac-sim/projects:/home/red0orange/projects:rw \
    isaac-sim-red0orange

docker exec -it --user red0orange isaac-sim-red0orange /isaac-sim/runheadless.native.sh -v
docker exec -it --user red0orange isaac-sim-red0orange /bin/zsh


### === final
docker run --name isaac-sim-red0orange-v3 --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v /data/huangdehao:/home/red0orange/data:rw \
    -v ~/docker/isaac-sim/projects:/home/red0orange/projects:rw \
    --cpus=8 \
    isaac-sim-red0orange-v3
docker container start isaac-sim-red0orange-v3
docker exec -it isaac-sim-red0orange-v3 sudo /usr/sbin/service ssh start
docker container update isaac-sim-red0orange-v3 --cpus="16"