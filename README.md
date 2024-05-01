# AlphaRepair

This repository is used to replicate the experiments of article "Towards Effective Multi-Hunk Bug Repair: Detecting, Creating, Evaluating, and Understanding Indivisible Bugs" on AlphaRepair. 

The repository is modified from the paper "Less Training, More Repairing Please: Revisiting Automated Program Repair via Zero-shot Learning", which is published at ESEC/FSE 2022. And the source code is available [here](https://zenodo.org/record/6819444). 



## 1. Modification

In this repository, we made the following changes:

- Add `add_new_line` method call.
- Separate the generation patch and verification patch parts.
- Limit the generation of up to 5000 patches per suspicious location (`add_new_line` and `process_file` count separately).
- The whole process of fixing each bug is limited to 5h.



## 2. Environment

- JDK 1.8
- Python 3.10
- [Defects4J 2.0.0](https://github.com/rjust/defects4j)
- [CatenaD4J](https://github.com/universetraveller/CatenaD4J)
- Ubuntu 20.04
- CUDA Version: 11.6



## 3 Experiment Setup

- Timeout: 5h
- beam_width: 5
- Plausible patches number limit at one suspicious location: 5000



## 4 Excluded Bug(s)

> None



## 5. Installation

### 5.1 Create the docker image

Use the `Dockerfile` in `./Docker` to create the docker image.

```shell
docker build -t alpharepair-env .
```

This docker image includes **Defects4J**, **CatenaD4J**, **JDK 1.8**, and **Python 3.10**.

### 5.2 Create the container with GPU

AlphaRepair requires the use of GPU, otherwise generating the patch part would be very slow.

```shell
docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all --name alpharepair alpharepair-env /bin/bash
```

### 5.3 Clone the AlphaRepair repository

At the root of this container, we clone the AlphaRepair repository.

```shell
cd /
git clone https://github.com/give-to/AlphaRepairAPI.git
cd /AlphaRepairAPI && chmod +x *
```

### 5.4 Install Dependencies

After testing, we found that using **pip** in the Docker file might cause image creation to fail, so configuration inside the container is required.

```shell
# Preparing the environment
cd /AlphaRepairAPI/code
pip install -r requirements.txt

# Install Pytorch for CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

If you have another version of CUDA, you can refer https://pytorch.org/get-started/previous-versions/ to install the corresponding Pytorch version.



## 6. Quick Test

It takes several minutes to quickly test your installation. (**Note:** In quick test, the `ochiai.ranking.txt` in Chart-18-2 only contains one location! )

```shell
cp location/ochiai/chart/18/2short location/ochiai/chart/18/2
# Generating the patches
python3 runAlphaRepair.py --start_bug_index=0 --end_bug_index=0
# Validate the patches
python3 runAlphaRepair.py --start_bug_index=0 --end_bug_index=0 --validation=True
```

After finishing the repair, the results are in folders: `codebert_result`. The plausible patch will show `Success (Plausible Patch)`.



## 7. Experiment Reproduction

It may take about **22 days** to finish the entire experiment. The main script is `runAlphaRepair.py`, it reads the list of bugs in the `buglist` file and fixes them sequentially. You can modify `buglist` to determine the bugs to be fixed.

```shell
cp location/ochiai/chart/18/2bak location/ochiai/chart/18/2
# Generating the patches
python3 runAlphaRepair.py --start_bug_index=0 --end_bug_index=104
# Validate the patches
python3 runAlphaRepair.py --start_bug_index=0 --end_bug_index=104 --validation=True
```

After finishing the repair, the results are in folders: `codebert_result`. 



**Note:** In the patch generation phase, the patches will be saved in the `store_changes` folder, and the time used to generate the patches will be saved in the `time_info` folder. **Do not delete these files to ensure the normal operation of patch validation.** The results of the patch validation will be in the `codebert_result` folder.



## 8. Structure of the Directories

```
 |--- README.md               :  User guidance
 |--- code                    :  Source code
 |------ C4J_results          :  Results about CatenaD4J by AlphaRepair
 |--------- codebert_result   :  Validation results for all patches
 |--------- store_changes     :  The process information of the patch generation
 |--------- time_info         :  The total time to generate the patch for each bug
 |------ codebert-base-mlm    :  The model used in AlphaRepair
 |------ location             :  Bug positions localized with GZoltar
 |--- correct_patches         :  Generated patches about Defects4J by AlphaRepair
```
