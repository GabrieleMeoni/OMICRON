# Description of submitted files
- submission_README.md: this file
- data: folder containing the labelled data
- environment.yml: Updated environment file (Python 3.10 required for hardware implementation). But installation of other hardware dependencies is still required.
- notebooks/hyperparameter_tuning.py: script to tune the hyperparameters of the CNN
- notebooks/model_mobilenet_v2.onnx and notebooks/model_mobilenet_v2.pth: trained model_mobilenet_v2 models for hardware (ONNX and PyTorch formats)
- notebooks/myriad_run_all.ipynb: notebook to run the model on the hardware
- utils/check_classification.py: script to check if the same image halves have been classified differently
- utils/classification_script.py: Since some issues were noticed with notebooks/Label_image_GUI.ipynb, this script was used to classify the images in some cases.