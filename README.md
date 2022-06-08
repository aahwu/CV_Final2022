# CV_Final2022
Final project for Computer Vision class in 2022

## The directory architecture:

    |---CV22S_Ganzin
        |---dataset
        |---script 

## eval.py
Evaluation for S1 ~ S4

## utils.py
Visualization for a specific video fragment

## main.py
For execute preprocessing.py, deepVOG.py, starburst.py

Should be runned at path "...\script"

## preprocessing.py
Preprocessing images

It will generate the folder "public/Sx_preimage" and write images to the corresponing folders "public/Sx_preimage/xx"

## deepVOG.py
deepVOG model

It will generate the folder "public/Sx_solution" and write label images to the corresponing folders "public/Sx_solution/xx"

environment: https://github.com/pydsgz/DeepVOG

## starburst.py
deepeye model & starburst algorithm

It will generate the folder "public/Sx_solution" and write label images to the corresponing folders "public/Sx_solution/xx"

environment: https://github.com/Fjaviervera/DeepEye

## DeepVOG_model.py & DeepVOG_weight.h5
Necessary files for running deepVOG.py

## deepeye.py & models
Necessary files for running starburst.py

## test_if_model_work.py & awesome.py
For testing