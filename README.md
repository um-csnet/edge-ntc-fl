# NTC-FL-Edge-XAI
Paper: Leveraging Federated Learning and XAI for Private and Lightweight Edge Training In Network Traffic Classification

# Deployed on:
## FL Server
* HP Pavilion 14
* Ryzen 5 (8 Core CPU)
* 16GB RAM
* 100 GB SSD Storage

## FL Client
* Nvidia Jetson Nano
* Quad-core ARM A57 CPU
* 128-core Maxwell GPU
* 4 GB RAM
* 64 GB eMMC Storage

## Software
* Tensorflow v2.6.0
* Flower 1.14.0
* Keras v2.11.0
* DeepSHAP v0.41.0

## Deep Learning Technique
1. MLP
2. 1D-CNN

## To deploy:
1. Download dataset here: https://www.unb.ca/cic/datasets/vpn.html and put in folder and run ISCX-VPN2016-pre-processing-v2.ipynb & ISCX-VPN2016-pre-processing_combine.ipynb script from Preprocessing folder
2. Put the processed raw data into /content/DATA and run preprocessraw.py script from Preprocessing folder
3. run the script from Centralized_FL_Experiment folder to run first experiment train the initial model
4. run the script from XAI_Experiment folder to perform feature selection and run the second experiment
5. run the evaluate.py script to evaluate model performance

For any inquiries you can email [azizi.mohdariffin@gmail.com]
