# Whole-Side-Image-PreProcessing
Cell feature Extraction


Synopsis:-
Cell Feature Extraction App : A graphical user interface (GUI) for the Whole Side Image Processing And  Cell Feature Extraction in python using tkinter

TCGA-OV histopathology data download
Cancer studies done based on tissue images for a particular cancer type generates Whole Slide Images, which generates data that is several Gigabytes in size as they contain much biological information which can be better analyzed with the use of deep learning models. We used HistomicsTK, an open-source software, to download and process TCGA-OV histopathology data from the GDC Data Portal. The GDC Data Portal had 107 SVS image files for Ovarian Cancer. HistomicsTK is a Python, and REST API developed for the analysis of histopathology whole-slide images in association with clinical and genomic data. HistomicsTK provides algorithms for fundamental image analysis tasks such as color normalization, color deconvolution, cell-nuclei segmentation, and feature extraction. HistomicsTK uses the `large_image` library to read and various microscopy image formats.

We modified two python scripts that were a part of the HistomicsTK package according to our datasets which were SVS files. This was done in two parts.
	One part was to split a whole image into different tiles so as to extract features from each cell of the tissue image. For this, the input is the TCGA.svs image and the output is the split images obtained from the TCGA.svs image.
	The second part was to extract cell features of each cell for which the input was each split image and the output was the ten features for each cell in the split images for the whole tissue slide.




Refer This Image :-

https://github.com/satyam8484/Whole-Side-Image-PreProcessing/blob/master/Fig_1.jpg









Analysis of Nuclear Morphology from Archived Histopathology Images
As outlined in  using our previously developed image analysis algorithms and pipeline  automated image analysis was carried out and ten types of cell-level features from tissue images were extracted following the three main steps: 1) nuclei segmentation, 2) cell-level feature measurement, and 3) aggregation of cell-level measurements into patient-level statistics.
In Step 1, the nuclei of all cells in the image are automatically segmented based on our previous workflow .
In Step 2, ten types of cell-level features were extracted, including seven types of morphological and spatial traits and three types of pixel traits in the RGB color space. The seven types of morphological and spatial features of cell nuclei were: major axis length (Major_Axis), minor axis length (Minor_Axis), the ratio of major to minor axis length (Ratio), nuclear area (Area), mean distance to neighboring cells (Mean_Distance), maximum distance to neighboring cells (Max_Distance), and minimum distance to neighboring cells (Min_Distance). The seven types of morphological and spatial features of cell nuclei can be summarized as nucleic area (Area), nucleic shape (Major_Axis, Minor_Axis, and Ratio), and cell density (Mean_Distance, Max_Distance and Min_Distance).
In Step 3, 5-bin histogram and five distribution statistics (i.e. mean, standard deviation or S.D., skewness, kurtosis, and entropy) were calculated for each of the ten types of morphological features to aggregate the measurements over the whole slide image. Thus for each type of feature, ten types of morphological features. measurements (i.e. five histogram bins and five distribution statistics) were generated and 100 image features were generated in total for the ten types of morphological features.




Installation:-

OS Used : Linux
Programming Language: Python 3.7
ForntEnd : tkinter

In Linux
# Run The Following Linux Commands

sudo apt-get install python3-tk

sudo pip3 install large_image

sudo pip3 install openslide-python

sudo pip3 install pandas

sudo pip3 install matplotlib

sudo pip3 install scikit-image

sudo pip3 install histomicstk

Check installation :-











