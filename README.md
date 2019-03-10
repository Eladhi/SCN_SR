# SCN - Super Resolution Seminar

This project is a final project in the course 236862 - Sparse and Redundant Representations and their Applications in Signal and Image Processing, of prof. Miki Elad (Technion).

The project is my implementation, with some extras, to the following paper: "Robust Single Image Super-Resolution via DeepNetworks with Sparse Prior”, by Liu et al., https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7466062.

## Paper Authors Code

The authors provided their code for forward pass only. It is used to compare the results.
The code and trained models appears in the folder 'downloaded'.

## Datasets

Training images should appear in: 'data/SR_training_datasets/<dataset_name>'
Test images should appear in: 'data/SR_test_datasets/<dataset_name>'

The dataset folder should contain 2 subfolder:
* original - with original (HR) images
* x<scaling_factor> - data for a specific scaling factor
	- <image_name> - original image
	- s_<image_name> - original image downsampled by the scaling factor
	- b_<image_name> - bicubic interpolation of the downsampled image (currently not necessary)

See the folder 'data/SR_test_datasets/Set5' for example.

For training, download SR benchmark (i.e. T91, BSDS200) and generate the downsampled images.

## CX loss

Contextual loss (https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/Contextual/) was also considered for this project. The initial results were unsatisfying. The code appears in 'CX' folder, but there was no optimization of it to perform better.