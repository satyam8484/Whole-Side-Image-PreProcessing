#!/usr/bin/env python
# coding: utf-8

# In[8]:


# !/usr/bin/env python
# coding: utf-8

# In[1]:

import histomicstk as htk
from tkinter import *
import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import statistics
import os
import statistics
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy
import pandas as pd

import large_image
import pandas as pd
import os
import os

import numpy as np
import matplotlib.pyplot as plt


# In[9]:


def get_ref_img():
    ref_image_file = ('https://data.kitware.com/api/v1/file/'
                      '57718cc28d777f1ecd8a883c/download')  # L1.png

    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]
    return im_reference




def Average(lst):
    try:
        return sum(lst) / len(lst)
    except:
        return 0


# In[3]:


def bin_size_calculation(l1, plot_fol_path, col_name):
    ## Calculating Bin Range ##
    MAL_list = l1
    df = pd.DataFrame(MAL_list)

    bin_size = pd.qcut(df[0], q=5, retbins=True, duplicates='drop')

    data = bin_size[1]
    data = data.tolist()

    # print(len(MAL_list))
    plt.figure()
    plt.hist(MAL_list, bins=data, rwidth=0.95)

    plt.savefig(plot_fol_path + "/Histogram_for_" + col_name)

    # print(data)
    # For Writing to Output File
    data_1 = data[1:]

    new = zip(data, data_1)
    new1 = list(new)
    return new1


def cell_Feature(out_path_split_image,pgbar,terminal):
    # In[4]:

    pgbar.config(value=70)
    # give path of input folder which contain all Folder(which contain splitted images) for each svs file
    path_of_input_folder = out_path_split_image

    folder_list = os.listdir(path_of_input_folder)
    print(len(folder_list))
    fol_cnt = 0
    flag = 0

    output_file_path = path_of_input_folder + "/" + path_of_input_folder.split("/")[-1] + "_OUTPUT.txt"
    # print(output_file_path)

    try:
        writer_1 = open(output_file_path, "x")
    except:
        flag = 1
        print("File is Already exists do you want continue Or YOU CAN DELETE IT MANUALLY")
        choice = input("Y/N \t")

        if (choice.lower() == "y"):
            writer_1 = open(output_file_path, "a")
        else:
            raise

    if (flag == 0):
        writer_1.write(
            "file Name\t" + "Average_Major Axis Length\t" + "Average_Minor Axis Length\t" + "Average_ratio of major to minor axis length\t" + "Average_Nuclear Area\t" + "Average_minimum distance to neighboring cells\t" + "Average_maximum distance to neighboring cells\t" + "Average_mean distance to neighboring cells" + "\t" + "R_Average\t" + "G_Average\t" + "B_Average\t")
        writer_1.write(
            "variance_Major Axis Length\t" + "variance_Minor Axis Length\t" + "variance_ratio of major to minor axis length\t" + "variance_Nuclear Area\t" + "variance_minimum distance to neighboring cells\t" + "variance_maximum distance to neighboring cells\t" + "variance_mean distance to neighboring cells" + "\t" + "variance_R\t" + "variance_G\t" + "variance_B\t")
        writer_1.write(
            "stdev_Major Axis Length\t" + "stdev_Minor Axis Length\t" + "stdev_ratio of major to minor axis length\t" + "stdev_Nuclear Area\t" + "stdev_minimum distance to neighboring cells\t" + "stdev_maximum distance to neighboring cells\t" + "stdev_mean distance to neighboring cells" + "\t" + "stdev_R\t" + "stdev_G\t" + "stdev_B\t")
        writer_1.write(
            "skewness_Major Axis Length\t" + "skewness_Minor Axis Length\t" + "skewness_ratio of major to minor axis length\t" + "skewness_Nuclear Area\t" + "skewness_minimum distance to neighboring cells\t" + "skewness_maximum distance to neighboring cells\t" + "skewness_mean distance to neighboring cells" + "\t" + "skewness_R\t" + "skewness_G\t" + "skewness_B\t")
        writer_1.write(
            "kurtosis_Major Axis Length\t" + "kurtosis_Minor Axis Length\t" + "kurtosis_ratio of major to minor axis length\t" + "kurtosis_Nuclear Area\t" + "kurtosis_minimum distance to neighboring cells\t" + "kurtosis_maximum distance to neighboring cells\t" + "kurtosis_mean distance to neighboring cells" + "\t" + "kurtosis_R\t" + "kurtosis_G\t" + "kurtosis_B\t")
        writer_1.write(
            "entropy_Major Axis Length\t" + "entropy_Minor Axis Length\t" + "entropy_ratio of major to minor axis length\t" + "entropy_Nuclear Area\t" + "entropy_minimum distance to neighboring cells\t" + "entropy_maximum distance to neighboring cells\t" + "entropy_mean distance to neighboring cells" + "\t" + "entropy_R\t" + "entropy_G\t" + "entropy_B" + "\t")
        writer_1.write(
            "Major Axis Length Bin 1" + "\t" + "Major Axis Length Bin 2" + "\t" + "Major Axis Length Bin 3" + "\t" + "Major Axis Length Bin 4" + "\t" + "Major Axis Length Bin 5" + "\t")
        writer_1.write(
            "Minor Axis Length Bin 1" + "\t" + "Minor Axis Length Bin 2" + "\t" + "Minor Axis Length Bin 3" + "\t" + "Minor Axis Length Bin 4" + "\t" + "Minor Axis Length Bin 5" + "\t")
        writer_1.write(
            "ratio of major to minor axis length Bin 1" + "\t" + "ratio of major to minor axis length Bin 2" + "\t" + "ratio of major to minor axis length Bin 3" + "\t" + "ratio of major to minor axis length Bin 4" + "\t" + "ratio of major to minor axis length Bin 5" + "\t")
        writer_1.write(
            "Nuclear Area Bin 1" + "\t" + "Nuclear Area Bin 2" + "\t" + "Nuclear Area Bin 3" + "\t" + "Nuclear Area Bin 4" + "\t" + "Nuclear Area Bin 5" + "\t")
        writer_1.write(
            "minimum distance to neighboring cells Bin 1" + "\t" + "minimum distance to neighboring cells Bin 2" + "\t" + "minimum distance to neighboring cells Bin 3" + "\t" + "minimum distance to neighboring cells Bin 4" + "\t" + "minimum distance to neighboring cells Bin 5" + "\t")
        writer_1.write(
            "maximum distance to neighboring cells Bin 1" + "\t" + "maximum distance to neighboring cells Bin 2" + "\t" + "maximum distance to neighboring cells Bin 3" + "\t" + "maximum distance to neighboring cells Bin 4" + "\t" + "maximum distance to neighboring cells Bin 5" + "\t")
        writer_1.write(
            "mean distance to neighboring cells Bin 1" + "\t" + "mean distance to neighboring cells Bin 2" + "\t" + "mean distance to neighboring cells Bin 3" + "\t" + "mean distance to neighboring cells Bin 4" + "\t" + "mean distance to neighboring cells Bin 5" + "\t")
        writer_1.write("R Bin 1" + "\t" + "R Bin 2" + "\t" + "R Bin 3" + "\t" + "R Bin 4" + "\t" + "R Bin 5" + "\t")
        writer_1.write("G Bin 1" + "\t" + "G Bin 2" + "\t" + "G Bin 3" + "\t" + "G Bin 4" + "\t" + "G Bin 5" + "\t")
        writer_1.write("B Bin 1" + "\t" + "B Bin 2" + "\t" + "B Bin 3" + "\t" + "B Bin 4" + "\t" + "B Bin 5" + "\t")
        writer_1.write("\n")

    for fol in folder_list:

        fol_path = path_of_input_folder + "/" + fol

        if (not (os.path.isdir(fol_path))):
            continue

        writer_1.write(fol + "\t")

        # path of PLOT OUTPU FOLDER
        plot_fol_path = fol_path + "_PLOT_OUTPUT"

        try:
            os.mkdir(plot_fol_path)
        except:
            print(' Already Exists ')

        print(fol_path)
        print(plot_fol_path)

        # Path of input folder which contain all splitted images
        path = fol_path

        # path of that txt file where we will Store output
        out_file_path = path + "_OUTPUT.txt"

        print(out_file_path)
        terminal.insert(END,out_file_path+"\n")

        try:
            list_cont = os.listdir(path)
        except:
            continue

        cnt = 0

        c1 = fol
        MAL_list = []
        MiAL_List = []
        ratio_maj_min_list = []
        Nuclear_area_list = []
        min_dist_list = []
        max_dist_list = []
        mean_dist_list = []
        r_col = []
        g_col = []
        b_col = []

        print(len(MAL_list), "LIST LENGTH")

        # PROCESSING ON EACH IMAGE IN FOLDER
        for img_part in list_cont:
            ##print(img_part)
            input_image_file = path + "/" + img_part
            out_file_path = out_file_path

            if (cnt == 0):
                try:
                    fw = open(out_file_path, "x")
                except:
                    print("OVERWRIITING ")
                    if os.path.exists(out_file_path):
                        os.remove(out_file_path)
                        print("OLD file Removed")
                    else:
                        print("The file does not exist")

                    # fw=open(out_file_path,"w")
                    # fw.write("")
                    # fw.close()

            cnt += 1
            im_input = skimage.io.imread(input_image_file)[:, :, :3]

            # plt.imshow(im_input)
            # _ = plt.title('Input Image', fontsize=16)

            ########  PART 2 ########

            # Load reference image for normalization

            if (fol_cnt == 0):
                ##print("AAAAAAAAAAAAAAAAAAAA")

                im_reference = get_ref_img()
                fol_cnt += 1
                ##print(fol_cnt)

            ##print(im_reference.shape)
            # get mean and stddev of reference image in lab space
            mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

            # perform reinhard color normalization
            im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)

            ##print(im_reference.shape)
            ##print(im_nmzd.shape)

            ######### PART 2.1 ##########

            # create stain to color map
            stainColorMap = {
                'hematoxylin': [0.65, 0.70, 0.29],
                'eosin': [0.07, 0.99, 0.11],
                'dab': [0.27, 0.57, 0.78],
                'null': [0.0, 0.0, 0.0]
            }

            # specify stains of input image
            stain_1 = 'hematoxylin'  # nuclei stain
            stain_2 = 'eosin'  # cytoplasm stain
            stain_3 = 'null'  # set to null of input contains only two stains

            # create stain matrix
            W = np.array([stainColorMap[stain_1],
                          stainColorMap[stain_2],
                          stainColorMap[stain_3]]).T

            # perform standard color deconvolution
            im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains

            ##########  PART 3 ############

            fw = open(out_file_path, "a")

            # get nuclei/hematoxylin channel
            im_nuclei_stain = im_stains[:, :, 0]

            # segment foreground
            foreground_threshold = 60

            im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
                im_nuclei_stain < foreground_threshold)

            # run adaptive multi-scale LoG filter
            min_radius = 10
            max_radius = 15

            im_log_max, im_sigma_max = htk.filters.shape.cdog(
                im_nuclei_stain, im_fgnd_mask,
                sigma_min=min_radius * np.sqrt(2),
                sigma_max=max_radius * np.sqrt(2)
            )

            # detect and segment nuclei using local maximum clustering
            local_max_search_radius = 10

            im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
                im_log_max, im_fgnd_mask, local_max_search_radius)

            # filter out small objects
            min_nucleus_area = 80

            im_nuclei_seg_mask = htk.segmentation.label.area_open(
                im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

            # compute nuclei properties
            objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

            # print ('Number of nuclei = ', len(objProps))

            # print(cnt,"Hello")
            if (cnt == 1):
                # print("Hello ")
                fw.write(
                    "Major Axis Length\t" + "Minor Axis Length\t" + "ratio of major to minor axis length\t" + "Nuclear Area\t" + "minimum distance to neighboring cells\t" + "maximum distance to neighboring cells\t" + "mean distance to neighboring cells\t" + "R\t" + "G\t" + "B\t" + "\n")

            # PROCESSING ON EACH CELLS of AN IMAGE
            for i in range(len(objProps) - 1):
                # print(i)
                MAL = objProps[i].major_axis_length
                MiAL = objProps[i].minor_axis_length

                # Calculate Ratio:-
                dec_ele_mal = MAL * 10 ** 9
                dec_ele_minor = MiAL * 10 ** 9

                GCF_MAL = dec_ele_mal / 2
                GCF_Mial = dec_ele_minor / 2

                ###fw.write(str(objProps[i].centroid)+"\t")
                fw.write(str(MAL) + "\t")
                fw.write(str(MiAL) + "\t")
                # fw.write(str(GCF_MAL)+" to "+str(GCF_Mial)+"\t")

                try:
                    ratio_val = GCF_MAL / GCF_Mial
                except:
                    ratio_val = 0

                fw.write(str(ratio_val) + "\t")

                nu_area = objProps[i].area
                fw.write(str(nu_area) + "\t")

                c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
                width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
                height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1

                cur_bbox = {
                    "type": "rectangle",
                    "center": c,
                    "width": width,
                    "height": height,
                }

                # plt.plot(c[0], c[1], 'g+')
                # print(width,height)
                mrect = mpatches.Rectangle([c[0] - 0.5 * width, c[1] - 0.5 * height],
                                           width, height, fill=False, ec='g', linewidth=2)
                # plt.gca().add_patch(mrect)

                ###  For Finding Minimum Distance And Maximum Distance

                dist_list = []

                x1, y1 = objProps[i].centroid
                dist_list = []
                ##print(objProps[i].centroid)

                for j in range(len(objProps)):
                    if (i != j):
                        x2, y2 = objProps[j].centroid
                        dist = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                        # print("Distance Bet two Cell",dist)
                        dist_list.append(dist)

                # print(dist_list)
                ##print("Minimum distance to cell",min(dist_list))
                ##print("Maximum distance to cell",max(dist_list))
                ##print("Mean Distance to cell",statistics.mean(dist_list))

                # RGB Color traits
                x = int(objProps[i].centroid[0])
                y = int(objProps[i].centroid[1])
                # print(im_input.shape)
                r, g, b = im_input[x, y]

                r = int(r)
                g = int(g)
                b = int(b)

                # print(type(MAL))
                # print(type(r))
                r_col.append(r)
                g_col.append(g)
                b_col.append(b)

                min_dist1 = min(dist_list)
                max_dist1 = max(dist_list)

                mean_dist1 = statistics.mean(dist_list)

                fw.write(str(min_dist1) + "\t")
                fw.write(str(max_dist1) + "\t")
                fw.write(str(mean_dist1) + "\t")

                fw.write(str(r) + "\t")
                fw.write(str(g) + "\t")
                fw.write(str(b) + "\n")

                MAL_list.append(MAL)
                MiAL_List.append(MiAL)

                ratio_maj_min_list.append(ratio_val)

                Nuclear_area_list.append(nu_area)

                min_dist_list.append(min_dist1)
                max_dist_list.append(max_dist1)
                mean_dist_list.append(mean_dist1)

        fw.close()

        print(len(MAL_list), "LIST LENGTH")

        # Calculate Average

        MAL_avg = Average(MAL_list)
        MiAL_avg = Average(MiAL_List)
        ratio_maj_min_avg = Average(ratio_maj_min_list)
        Nuclear_area_avg = Average(Nuclear_area_list)
        min_dist_avg = Average(min_dist_list)
        max_dist_avg = Average(max_dist_list)
        mean_dist_avg = Average(mean_dist_list)

        r_avg = Average(r_col)
        g_avg = Average(g_col)
        b_avg = Average(b_col)

        """ 
        print(MAL_avg)
        print(MiAL_avg)
        print(ratio_maj_min_avg)
        print(Nuclear_area_avg)
        print(min_dist_avg)
        print(max_dist_avg)
        print(mean_dist_avg)
        """

        # Calculate variance
        MAL_variance = statistics.variance(MAL_list)
        MiAL_variance = statistics.variance(MiAL_List)
        ratio_maj_min_variance = statistics.variance(ratio_maj_min_list)
        Nuclear_area_variance = statistics.variance(Nuclear_area_list)
        min_dist_variance = statistics.variance(min_dist_list)
        max_dist_variance = statistics.variance(max_dist_list)
        mean_dist_variance = statistics.variance(mean_dist_list)

        # print(r_col)
        r_variance = statistics.variance(r_col)
        g_variance = statistics.variance(g_col)
        b_variance = statistics.variance(b_col)

        # Calculate stdev

        MAL_stdev = statistics.stdev(MAL_list)
        MiAL_stdev = statistics.stdev(MiAL_List)
        ratio_maj_min_stdev = statistics.stdev(ratio_maj_min_list)
        Nuclear_area_stdev = statistics.stdev(Nuclear_area_list)
        min_dist_stdev = statistics.stdev(min_dist_list)
        max_dist_stdev = statistics.stdev(max_dist_list)
        mean_dist_stdev = statistics.stdev(mean_dist_list)

        r_stdev = statistics.stdev(r_col)
        g_stdev = statistics.stdev(g_col)
        b_stdev = statistics.stdev(b_col)

        # Calculate skewness

        MAL_skew = skew(MAL_list)
        MiAL_skew = skew(MiAL_List)
        ratio_maj_min_skew = skew(ratio_maj_min_list)
        Nuclear_area_skew = skew(Nuclear_area_list)
        min_dist_skew = skew(min_dist_list)
        max_dist_skew = skew(max_dist_list)
        mean_dist_skew = skew(mean_dist_list)

        r_skew = skew(r_col)
        g_skew = skew(g_col)
        b_skew = skew(b_col)

        # Calculate kurtosis

        MAL_kurt = kurtosis(MAL_list)
        MiAL_kurt = kurtosis(MiAL_List)
        ratio_maj_min_kurt = kurtosis(ratio_maj_min_list)
        Nuclear_area_kurt = kurtosis(Nuclear_area_list)
        min_dist_kurt = kurtosis(min_dist_list)
        max_dist_kurt = kurtosis(max_dist_list)
        mean_dist_kurt = kurtosis(mean_dist_list)

        r_kurt = kurtosis(r_col)
        g_kurt = kurtosis(g_col)
        b_kurt = kurtosis(b_col)

        # Calculate entropy

        MAL_entropy = entropy(MAL_list)
        MiAL_entropy = entropy(MiAL_List)
        ratio_maj_min_entropy = entropy(ratio_maj_min_list)
        Nuclear_area_entropy = entropy(Nuclear_area_list)
        min_dist_entropy = entropy(min_dist_list)
        max_dist_entropy = entropy(max_dist_list)
        mean_dist_entropy = entropy(mean_dist_list)

        r_entropy = entropy(r_col)
        g_entropy = entropy(g_col)
        b_entropy = entropy(b_col)

        ## Calculating Bin Range ##

        df = pd.DataFrame(MAL_list)

        bin_size = pd.qcut(df[0], q=5, retbins=True, duplicates='drop')

        data = bin_size[1]
        data = data.tolist()

        # print(len(MAL_list))
        plt.figure()
        plt.hist(MAL_list, bins=data, rwidth=0.95)

        plt.savefig(plot_fol_path + "/Histogram_Rep_")

        # print(data)
        # For Writing to Output File
        data_1 = data[1:]

        new = zip(data, data_1)
        new = list(new)

        # df1=pd.qcut(df[0], q=5)
        # print(df1.value_counts())

        #### writing result ####

        writer_1.write(str(MAL_avg) + "\t")
        writer_1.write(str(MiAL_avg) + "\t")
        writer_1.write(str(ratio_maj_min_avg) + "\t")
        writer_1.write(str(Nuclear_area_avg) + "\t")
        writer_1.write(str(min_dist_avg) + "\t")
        writer_1.write(str(max_dist_avg) + "\t")
        writer_1.write(str(mean_dist_avg) + "\t")

        writer_1.write(str(r_avg) + "\t")
        writer_1.write(str(g_avg) + "\t")
        writer_1.write(str(b_avg) + "\t")

        writer_1.write(str(MAL_variance) + "\t")
        writer_1.write(str(MiAL_variance) + "\t")
        writer_1.write(str(ratio_maj_min_variance) + "\t")
        writer_1.write(str(Nuclear_area_variance) + "\t")
        writer_1.write(str(min_dist_variance) + "\t")
        writer_1.write(str(max_dist_variance) + "\t")
        writer_1.write(str(mean_dist_variance) + "\t")

        writer_1.write(str(r_variance) + "\t")
        writer_1.write(str(g_variance) + "\t")
        writer_1.write(str(b_variance) + "\t")

        writer_1.write(str(MAL_stdev) + "\t")
        writer_1.write(str(MiAL_stdev) + "\t")
        writer_1.write(str(ratio_maj_min_stdev) + "\t")
        writer_1.write(str(Nuclear_area_stdev) + "\t")
        writer_1.write(str(min_dist_stdev) + "\t")
        writer_1.write(str(max_dist_stdev) + "\t")
        writer_1.write(str(mean_dist_stdev) + "\t")

        writer_1.write(str(r_stdev) + "\t")
        writer_1.write(str(g_stdev) + "\t")
        writer_1.write(str(b_stdev) + "\t")

        writer_1.write(str(MAL_skew) + "\t")
        writer_1.write(str(MiAL_skew) + "\t")
        writer_1.write(str(ratio_maj_min_skew) + "\t")
        writer_1.write(str(Nuclear_area_skew) + "\t")
        writer_1.write(str(min_dist_skew) + "\t")
        writer_1.write(str(max_dist_skew) + "\t")
        writer_1.write(str(mean_dist_skew) + "\t")

        writer_1.write(str(r_skew) + "\t")
        writer_1.write(str(g_skew) + "\t")
        writer_1.write(str(b_skew) + "\t")

        writer_1.write(str(MAL_kurt) + "\t")
        writer_1.write(str(MiAL_kurt) + "\t")
        writer_1.write(str(ratio_maj_min_kurt) + "\t")
        writer_1.write(str(Nuclear_area_kurt) + "\t")
        writer_1.write(str(min_dist_kurt) + "\t")
        writer_1.write(str(max_dist_kurt) + "\t")
        writer_1.write(str(mean_dist_kurt) + "\t")

        writer_1.write(str(r_kurt) + "\t")
        writer_1.write(str(g_kurt) + "\t")
        writer_1.write(str(b_kurt) + "\t")

        writer_1.write(str(MAL_entropy) + "\t")
        writer_1.write(str(MiAL_entropy) + "\t")
        writer_1.write(str(ratio_maj_min_entropy) + "\t")
        writer_1.write(str(Nuclear_area_entropy) + "\t")
        writer_1.write(str(min_dist_entropy) + "\t")
        writer_1.write(str(max_dist_entropy) + "\t")
        writer_1.write(str(mean_dist_entropy) + "\t")

        writer_1.write(str(r_entropy) + "\t")
        writer_1.write(str(g_entropy) + "\t")
        writer_1.write(str(b_entropy) + "\t")

        for f, s in new:
            bin_range = str(f) + " - " + str(s)
            writer_1.write(bin_range + "\t")

        col_list = [MiAL_List, ratio_maj_min_list, Nuclear_area_list, min_dist_list, max_dist_list, mean_dist_list,
                    r_col, g_col, b_col]
        names = ['Minor_Axis_Length', 'ratio of major to minor axis length', 'Nuclear Area',
                 'minimum distance to neighboring cells', 'maximum distance to neighboring cells',
                 'mean distance to neighboring cells', 'R', 'G', 'B']
        for i in range(len(col_list)):
            # print(list_item,plot_fol_path)
            bin_list = bin_size_calculation(col_list[i], plot_fol_path, names[i])
            # print(type(bin_list))
            for f, s in bin_list:
                bin_range = str(f) + " - " + str(s)
                writer_1.write(bin_range + "\t")

        writer_1.write("\n")

    writer_1.close()


# In[ ]:


#########################################

# In[5]:

def split_image(path,pgbar,terminal):
    # path of that folder which contain all Folder(which contain .svs files)
    input_fol_path = path

    # path of that folder where o/p will generate
    out_path = input_fol_path + "_OUTPUT"
    print(out_path)
    terminal.insert(END,out_path+"\n")
    pgbar.config(value=40)
    try:
        os.mkdir(out_path)
    except:
        print('Folder Already exists ')

    # In[6]:

    list_contents = os.listdir(input_fol_path)
    ct = 1
    for contents_1 in list_contents:

        fol_path = input_fol_path + "/" + contents_1

        if (os.path.isdir(fol_path)):
            list_fol_contents = os.listdir(fol_path)
        else:
            continue

        print(fol_path)
        terminal.insert(END,fol_path+"\n")

        ct += 1
        for contents in list_fol_contents:
            # print(contents)
            if (contents.endswith(".svs")):
                input_SVS_File_path = fol_path + "/" + contents
                print(input_SVS_File_path)
                terminal.insert(END,input_SVS_File_path+"\n")

                # wsi_path=input_SVS_File_path
                # ts = large_image.getTileSource(wsi_path)

                #### HERE WE ARE LOADING A WHOLE-SLIDE IMAGE  ####
                try:
                    wsi_path = input_SVS_File_path
                    ts = large_image.getTileSource(wsi_path)
                except:
                    print(wsi_path + "There is problem with this File it may be Corupted OR CHECK Path")
                    continue

                ##print(os.path.getsize(wsi_path))

                """

                THE tileIterator() FUNCTION PROVIDES A ITERATOR FOR SEQUENTIALLY ITERATING THROUGH 
                THE ENTIRE SLIDE OR A REGION OF INTEREST (ROI) WITHIN THE SLIDE AT ANY DESIRED RESOLUTION 
                IN A TILE-WISE FASHION.

                """

                # 2)  Getting Number Of Tiles
                num_tiles = 0

                # tile_means = []
                # tile_areas = []

                for tile_info in ts.tileIterator(
                        region=dict(left=5000, top=5000, width=20000, height=20000, units='base_pixels'),
                        scale=dict(magnification=20),
                        tile_size=dict(width=1000, height=1000),
                        tile_overlap=dict(x=50, y=50),
                        format=large_image.tilesource.TILE_FORMAT_PIL
                ):
                    """if num_tiles == 100:
                        print('Tile-{} = '.format(num_tiles))
                        display(tile_info)

                    im_tile = np.array(tile_info['tile'])
                    tile_mean_rgb = np.mean(im_tile[:, :, :3], axis=(0, 1))

                    tile_means.append( tile_mean_rgb )
                    tile_areas.append( tile_info['width'] * tile_info['height'] ) """

                    num_tiles += 1

                # slide_mean_rgb = np.average(tile_means, axis=0, weights=tile_areas)

                print('Number of tiles = {}'.format(num_tiles))
                # print('Slide mean color = {}'.format(slide_mean_rgb))

                # 3) CREATING AN FOLDER FOR EACH IMAGE AND SPLIT IT

                df_list = []
                ##print(num_tiles)
                tile_info = ""
                fol_name = input_SVS_File_path.split("/")[-1].strip(".svs")
                ##print(fol_name)

                image_path = out_path + "/" + fol_name
                extension = ".png"

                # Create a folder by the name of svs file name
                try:
                    os.mkdir(image_path)
                except:
                    print("Folder Already Exists !!! ")

                """

                THE getSingleTile() FUNCTION CAN BE USED TO DIRECTLY GET THE TILE AT A SPECIFIC POSITION 
                OF THE TILE ITERATOR

                """
                image_dir_path = image_path
                image_path = image_path + "/part"

                err = 0
                for i in range(1, num_tiles + 1):

                    img_path = image_path + str(i) + extension
                    ##print(img_path)
                    pos = i
                    ##print(pos)
                    tile_info = ts.getSingleTile(
                        tile_size=dict(width=1000, height=1000),
                        scale=dict(magnification=20),
                        tile_position=pos
                    )
                    ##print(tile_info)
                    # plt.imshow(tile_info['tile'])
                    try:
                        plt.imsave(img_path, tile_info['tile'])
                    except:
                        """name_fol=fol_path.split("/")[-1]
                        fol_path_new=fol_path.strip(name_fol)+"ERROR.txt"
                        fw_out=open(fol_path_new)
                        fw_out.write(fol_path_new+" ")
                        fol_path_new.close()"""
                        print("ERROR IN FOLDER " + fol_path)
                        err = 1
                        break
                        # print(tile_info['tile'].shape)
                if (err == 1):
                    try:
                        os.system("rm -r " + image_dir_path)
                    except:
                        pass
    print("No of folder ", ct)
    return out_path







