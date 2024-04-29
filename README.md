# AlphaRepair

This repository implements research paper "Less Training, More Repairing Please: Revisiting Automated Program Repair via Zero-shot Learning", which is published at ESEC/FSE 2022. The source code are modified from official [Zenodo repo](https://zenodo.org/record/6819444).

## 1. Environment

- JDK 1.8
- python 3.10
- [Defects4J 2.0.0](https://github.com/rjust/defects4j)
- [CatenaD4J](https://github.com/universetraveller/CatenaD4J)
- Ubuntu 20.04
- CUDA Version: 11.6



## 2. Installation

### 2.1 Create the docker image

Use the `Dockerfile` in `./Docker` to create the docker image.

```shell
docker build -t alpharepair-env .
```

This docker image includes **Defects4J**, **CatenaD4J**, **JDK 1.8**, and **Python 3.10**.

### 2.2 Create the container with GPU

AlphaRepair requires the use of GPU, otherwise generating the patch part would be very slow.

```shell
docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all --name alpharepair alpharepair-env /bin/bash
```

### 2.3 Clone the AlphaRepair repository

At the root of this container, we clone the AlphaRepair repository.

```shell
cd /
git clone https://github.com/give-to/AlphaRepairAPI.git
```

### 2.4 Install Dependencies

After testing, we found that using **pip** in the Docker file might cause image creation to fail, so configuration inside the container is required.

```shell
# Preparing the environment
cd /AlphaRepairAPI/code
pip install -r requirements.txt

# Install Pytorch for CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

If you have another version of CUDA, you can refer https://pytorch.org/get-started/previous-versions/ to install the corresponding Pytorch version.



## 3. Quick Test

Run the test experiment to ensure your environment is correct. The part of generating the patches will take between 80 and 110 minutes of your time. These two commands take a maximum of 5 hours.

```shell
# Generating the patches
python3 runAlphaRepair.py --start_bug_index=0 --end_bug_index=0
# Validate the patches
python3 runAlphaRepair.py --start_bug_index=0 --end_bug_index=0 --validation=True
```



## 4. Repeat whole experiments

```shell
# Generating the patches
python3 runAlphaRepair.py --start_bug_index=0 --end_bug_index=104
# Validate the patches
python3 runAlphaRepair.py --start_bug_index=0 --end_bug_index=104 --validation=True
```



## 5. Usage

The main script is `runAlphaRepair.py`, it reads the list of bugs in the `buglist` file and fixes them sequentially. It accept 3 parameters:

- **start_bug_index:** bug start index in `buglist` file. (The default is 0)
- **end_bug_index:** bug end index in `buglist` file. (The default is 0)
- **validation:** Whether to validate. (The default is False)

This repository separates patch generation and patch validation. 

1. To generate the patches

   ```shell
   python3 runAlphaRepair.py --start_bug_index=<start_bug_index> --end_bug_index=<end_bug_index>
   ```

2. To validate the patches

   ```shell
   python3 runAlphaRepair.py --start_bug_index=<start_bug_index> --end_bug_index=<end_bug_index> --validation=True
   ```


In the patch generation phase, the patches will be saved in the `store_changes` folder, and the time used to generate the patches will be saved in the `time_info` folder. **Do not delete these files to ensure the normal operation of patch validation.** The results of the patch validation will be in the `codebert_result` folder.



## 6. Structure of the Directories

```
 |--- README.md               :  user guidance
 |--- code                    :  source code
 |------ C4J_results          :  results about CatenaD4J by AlphaRepair
 |--------- codebert_result   :  Validation results for all patches
 |--------- store_changes     :  The process information of the patch generation
 |--------- time_info         :  The total time to generate the patch for each bug
 |--- correct_patches         :  Generated patches about Defects4J by AlphaRepair
```
