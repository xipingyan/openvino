# Enable Sycl Runtime pipeline for OpenVINO GPU plugin
This is internal share development guide about enable SYCL Runtime pipeline.

# How to build
#### OpenVINO.

repos: https://github.com/xipingyan/openvino.git <br>
Branch: xp/sycl_opt_master <br>

```
git clone https://github.com/xipingyan/openvino.git
cd openvino
git checkout -b xp/sycl_opt_master origin/xp/sycl_opt_master
mkdir build && cd build
git submodule update --init
```

#### Dependencies:
1: oneAPI 2025 <br>
2: Upgrade rapidjson <br>
```
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson && mkdir build && cd build
cmake .. && make -j20
sudo make install
```

#### Build:
```
source /opt/intel/oneapi/setvars.sh
cd openvino && make build && cd build
# (-DENABLE_GPU_DEBUG_CAPS=ON -DENABLE_DEBUG_CAPS=ON) Only for print GPU debug log
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DENABLE_INTEL_CPU=OFF -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_GPU_DEBUG_CAPS=ON -DENABLE_DEBUG_CAPS=ON -DCMAKE_INSTALL_PREFIX=install ..
```
