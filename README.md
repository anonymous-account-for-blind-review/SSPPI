# SSPPI
---
A repo for "SSPPI: Cross-modality enhanced protein-protein interaction prediction from sequence and structure perspectives".


## Contents

* [Abstracts](#abstracts)
* [Requirements](#requirements)
   * [Download projects](#download-projects)
   * [Configure the environment manually](#configure-the-environment-manually)
   * [Docker Image](#docker-image)
* [Usages](#usages)
   * [Data preparation](#data-preparation)
   * [Training](#training)
   * [Pretrained models](#pretrained-models)
   * [Reproduce the results with single command](#reproduce-the-results-with-single-command)
* [Cancer cases](#cancer-cases)

## Abstracts
Recent advances have shown great promise in mining multi-modal protein knowledge for better protein-protein interaction (PPI) prediction by enriching the representation of proteins. However, existing solutions lack a comprehensive consideration of both local patterns and global dependencies in proteins, hindering the full exploitation of modal information. Additionally, the inherent disparities between modalities are often disregarded, which may lead to inferior modality complementarity effects. To address these issues, we propose a cross-modality enhanced PPI prediction method from the perspectives of protein Sequence and Structure modalities, namely `SSPPI`. In this framework, our main contribution is that we integrate both sequence and structural modalities of proteins and employ an alignment and fusion method between modalities to further generate more comprehensive protein representations for PPI prediction. Specifically, we design two modal representation modules (`Convformer` and `Graphormer`) tailored for protein sequence and structure modalities, respectively, to enhance the quality of modal representation. Subsequently, we introduce a `Cross-modality enhancer` module to achieve alignment and fusion between modalities, thereby generating more informative modal joint representations. Finally, we devise a `Cross-protein fusion` module to model residue interaction processes between proteins, thereby enriching the joint representation of protein pairs. Extensive experimentation on three benchmark datasets demonstrates that our proposed model surpasses all current state-of-the-art (SOTA) methods in PPI prediction performance.

![SSPPI architecture](https://github.com/bixiangpeng/SSPPI/blob/main/framework.png)


## Requirements

* ### Download projects

   Download the GitHub repo of this project onto your local server: `git clone https://github.com/bixiangpeng/SSPPI`


* ### Configure the environment manually

   Create and activate virtual env: `conda create -n SSPPI python=3.8 ` and `conda activate SSPPI`
   
   Install specified version of pytorch: ` conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
   
   Install specified version of PyG: ` conda install pyg==2.2.0 -c pyg`
   
   :bulb: Note that the operating system we used is `ubuntu 22.04` and the version of Anaconda is `23.3.1`.


* ### Docker Image

    We also provide the Dockerfile to build the environment, please refer to the Dockerfile for more details. Make sure you have Docker installed locally, and simply run following command:
   ```shell
   # Build the Docker image
   sudo docker build --build-arg env_name=HiSIF -t hisif-image:v1 .
   # Create and start the docker container
   sudo docker run --name hisif-con --gpus all -it hisif-image:v1 /bin/bash
   # Check whether the environment deployment is successful
   conda list 
   ```

  
##  Usages

* ### Data preparation
  There are three benchmark datasets were adopted in this project, including two binary classification datasets (`Yeast and Multi-species`) and a multi-class classification dataset (`Multi-class`).

   1. __Download processed data__
   
      The data file (`data.zip`) of these three datasets can be downloaded from this [link](https://pan.baidu.com/s/1KI4DrDVXInQaM5Wv1_0NSw?pwd=6shz). Uncompress this file to get a 'data' folder containing all the original data and processed data.
      
      🌳 Replacing the original 'data' folder by this new folder and then you can re-train or test our proposed model on Yeast, Multi-species or Multi-class.  
      
   3. __Customize your data__
      
      You can reprocess data or customize your own data by executing the following command：`python data_process.py`
      

* ### Training
  After processing the data, you can retrain the model from scratch with the following command:
  ```text
  For Yeast dataset:
    python my_main.py --datasetname yeast --output_dim 1

  For Multi-species dataset:
    python my_main.py --datasetname multi_species --output_dim 1 --identity any

  For Multi-class dataset:
    python my_main.py --datasetname multi_class --output_dim 7  

   ```
  
  Here is the detailed introduction of the optional parameters when running `my_main.py`:
   ```text
    --datasetname: The dataset name, specifying the dataset used for model training.
    --output_dim: The parameter for specifying the number of PPI categories in the dataset.
    --identity: The threshold of identity, specifying the multi-species dataset under this sequence identity.
    --device_id: The device, specifying the GPU device number used for training.
    --batch_size: The batch size, specifying the number of samples in each training batch.
    --epochs: The number of epochs, specifying the number of iterations for training the model on the entire dataset.
    --lr: The learning rate, controlling the rate at which model parameters are updated.
    --num_workers: This parameter is an optional value in the Dataloader, and when its value is greater than 0, it enables multiprocessing for data processing.
    --rst_path: The parameter for specifying the file saving path.  ```

* ### Pretrained models

   If you don't want to re-train the model, we provide pre-trained model parameters as shown below. 
<a name="pretrained-models"></a>
   | Datasets | Pre-trained models          | Description |
   |:-----------:|:-----------------------------:|:--------------|
   | Yeast    | [model](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/yeast/model.pkl) | The pretrained model parameters on the Yeast. |
   | Multi-species     | [model_01](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_01.pkl) &nbsp; , &nbsp; [model_10](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_10.pkl) &nbsp; , &nbsp; [model_25](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_25.pkl) &nbsp; , &nbsp; [model_40](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_40.pkl)  &nbsp; , &nbsp; [model_any](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_any.pkl)      | The Pretrained model parameters on the Multi-species under different sequence identities. |
   | Multi-class    | [model](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_class/model.pkl)   | The pretrained model parameters on the Multi-class dataset. |
  
   Based on these pre-trained models, you can perform PPI predictions by simply running the following command:
   ```text
    For Yeast dataset:
      python inference.py --datasetname yeast --output_dim 1
  
    For Multi-species dataset:
      python inference.py --datasetname multi_species --output_dim 1 --identity any
  
    For Multi-class dataset:
      python inference.py --datasetname multi_class --output_dim 7  

   ```

  * ### Reproduce the results with single command
   To facilitate the reproducibility of our experimental results, we have provided a Docker Image-based solution that allows for reproducing our experimental results on multiple datasets with just a single command. You can easily experience this function with the following simple command：
  ```text
  sudo docker run --name hisif-con --gpus all --shm-size=2g -v /your/local/path/HiSIF-DTA/:/media/HiSIF-DTA -it hisif-image:v1

  # docker run ：Create and start a new container based on the specified image.
  # --name : It specifies the name ("hisif-con") for the container being created. You can use this name to reference and manage the container later.
  # --gpus : It enables GPU support within the container and assigns all available GPUs to it. This allows the container to utilize the GPU resources for computation.
  # -v : This is a parameter used to map local files to the container,and it is used in the following format: `-v /your/local/path/HiSIF-DTA:/mapped/container/path/HiSIF-DTA`
  # -it : These options are combined and used to allocate a pseudo-TTY and enable interactive mode, allowing the user to interact with the container's command-line interface.
  # hisif-image:v1 : It is a doker image, builded from Dockerfile. For detailed build instructions, please refer to the `Requirements` section.
  ```
  :bulb: Please note that the above one-click run is only applicable for the inference process and requires you to pre-place all the necessary processed data and pretrained models in the correct locations on your local machine. If you want to train the model in the created Docker container, please follow the instructions below:
   ```text
   1. sudo docker run --name hisif-con --gpus all --shm-size=16g -v /your/local/path/HiSIF-DTA/:/media/HiSIF-DTA -it hisif-image:v1 /bin/bash
   2. cd /media/HiSIF-DTA
   3. python training_for_DTA.py --dataset davis --model TDNet
   ```
