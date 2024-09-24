# Bugs
### pyrender bug

```bash
sudo apt-get install -y python3-opengl
sudo apt-get install -y libosmesa6
```

- https://blog.csdn.net/qq_44354520/article/details/129345590
- https://blog.csdn.net/weixin_40437821/article/details/115200555
- https://blog.csdn.net/CCCDeric/article/details/129292944
- https://blog.csdn.net/guntangsanjing/article/details/127651381

- https://github.com/huggingface/deep-rl-class/issues/275

# [ICRA 2024] Language-Conditioned Affordance-Pose Detection in 3D Point Clouds

Official code for the ICRA 2024 paper "Language-Conditioned Affordance-Pose Detection in 3D Point Clouds". Our paper is currently available at [this arXiv](https://arxiv.org/pdf/2309.10911.pdf).

<img src="./assets/intro.png" width="500">

We address the task of language-driven affordance-pose detection in 3D point clouds. Our method simultaneously detect open-vocabulary affordances and
generate affordance-specific 6-DoF poses.

![image](./assets/method.png)
We present 3DAPNet, a new method for affordance-pose joint learning. Given the captured 3D point cloud of an object and a set of affordance labels conveyed through natural language texts, our objective is to jointly produce both the relevant affordance regions and the appropriate pose configurations that facilitate the affordances.


## Getting Started
We strongly encourage you to create a separate conda environment.
```
conda create -n affpose python=3.8
conda activate affpose
conda install pip
pip install -r requirements.txt
```

## Dataset
Our 3DAP dataset is available at [this drive folder](https://drive.google.com/drive/folders/1vDGHs3QZmmF2rGluGlqBIyCp8sPR4Yws?usp=sharing).

## Training
Current framework supports training on a single GPU. Followings are the steps for training our method with configuration file ```config/detectiondiffusion.py```.

* In ```config/detectiondiffusion.py```, change the value of ```data_dir``` to your downloaded data folder.
* Change other hyperparameters if needed.
* Assume you use the GPU 0, then run the following command to start training:

		CUDA_VISIBLE_DEVICES=0 python3 train.py --config ./config/detectiondiffusion.py

## Open-Vocabulary Testing
Executing the following command for testing of your trained model:

                CUDA_VISIBLE_DEVICES=0 python3 detect.py --config <your configuration file> --checkpoint <your  trained model checkpoint> --test_data <test data in the 3DAP dataset>

Note that we current generate 2000 poses for each affordance-object pair.
The guidance scale is currently set to 0.5. Feel free to change these hyperparameters according to your preference.

The result will be saved to a ```result.pkl``` file.

## Visualization
To visuaize the result of affordance detection and pose estimation, execute the following script:

                python3 visualize.py --result_file <your result pickle file>

Example of visualization:

<img src="./assets/visualization.png" width="500">

## Citation

If you find our work useful for your research, please cite:
```
@inproceedings{Nguyen2024language,
      title={Language-Conditioned Affordance-Pose Detection in 3D Point Clouds},
      author={Nguyen, Toan and Vu, Minh Nhat and Huang, Baoru and Van Vo, Tuan and Truong, Vy and Le, Ngan and Vo, Thieu and Le, Bac and Nguyen, Anh},
      booktitle = ICRA,
      year      = {2024}
  }
```
Thank you very much.