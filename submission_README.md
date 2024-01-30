# Description of submitted files
- submission_README.md: this file
- data: folder containing the labelled data
- environment.yml: Updated environment file (Python 3.10 required for hardware implementation). But installation of other hardware dependencies is still required.


- notebooks/hyperparameter_tuning.py: script to tune the hyperparameters of the CNN
- notebooks/models: folder containing the trained models
- notebooks/myriad_run.ipynb: notebook to run the model on the hardware for a single image and to convert a pytorch model to onnx
- notebooks/myriad_run_all.ipynb: notebook to run the model on the hardware for all images in the dataset
- notebooks/optuna_results.csv: results of the hyperparameter tuning
- notebooks/Train.ipynb: main notebook to train the model
- notebooks/result_plots: folder containing the plots of the training. Files are automatically saved here
- notebooks/plots: folder containing plots. Files here have been moved manually from notebooks/result_plots


- utils/check_classification.py: script to check if the same image halves have been classified differently
- utils/classification_script.py: Since some issues were noticed with notebooks/Label_image_GUI.ipynb, this script was used to classify the images in some cases.