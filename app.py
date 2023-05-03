import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os
from KMean import KMeans
import meanShift_optimal as MsO
import spectral_threshold as spct
from skimage.io import imread
import Region_Growing as RG
import rgb2luv as luv
import Thresholding as thresh


option=''

path='images'
st.set_page_config(page_title="Image segmentation",layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.sidebar:
    st.title('Upload an image')
    uploaded_file = st.file_uploader("", accept_multiple_files=False, type=['jpg','png','jpeg','webp'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        plt.imread(uploaded_file)
        image_path1=os.path.join(path,uploaded_file.name)
        st.title("Options")
        option = st.selectbox("",["Segmentation using K-means","Segmentation using mean shift","Optimized Thresholding","Spectral Thresholding","Region Growing","RGB to LUV","Manual & Otsu's Thresholding"])
        if option == "Segmentation using K-means":
            max_iter = st.slider(label="Max number of iterations",min_value=1, max_value=100, step=2)
            k = st.slider(label="clusters",min_value=1, max_value=5, step=1)
            
        if option == "Segmentation using mean shift":
            max_iter = st.slider(label="Max number of iterations",min_value=50, max_value=1000, step=50)
            
        if option == "Optimized Thresholding":
            option1 = st.selectbox("",["Global","Local"])
          
input_img, resulted_img = st.columns(2)
with input_img:
    if uploaded_file is not None:
            st.title("Input images")
            image = Image.open(uploaded_file)
            st.image(uploaded_file)
            # if (option == 'normalized cross correlations') or (option == 'Sum of Square Difference') or (option == 'Apply SIFT'):
            #     if second_image is not None: 
            #         image2 = Image.open(second_image)
            #         st.image(second_image)

with resulted_img:
    st.title("output image")

    if option == 'Segmentation using K-means':
        if uploaded_file is not None:     
            image=cv2.imread(image_path1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert to RGB
            X = image.reshape((-1,3))   #Reshape to (Npts, Ndim = 3)
            X = np.float32(X)
            km = KMeans(n_clus = k)
            km.fit(X,200)
            centers = km.getCentroids()
            clusters = km.getClusters()

            segmented_image = centers[clusters]

            segmented_image = segmented_image.reshape((image.shape))

            plt.imshow((segmented_image).astype(np.uint8))
            # plt.tick_params(labelleft=False, labelbottom=False, labelright=False, labeltop=False)
            plt.axis("off")
            plt.show()
            plt.savefig('./images/output/kmean.jpg')
            st.image('./images/output/kmean.jpg')
            
    if option == 'Segmentation using mean shift':
        if uploaded_file is not None:
            # st.title("output image")
            MsO.mean_shift_method(image_path1,max_iter)
            st.image('./images/output/mean_shift_out.png')
            
    if option == 'Optimized Thresholding':
        if uploaded_file is not None:
            out_img = MsO.Optimized_Thresholding(image_path1,option1)
            st.image(out_img)
    
    if option == "Spectral Thresholding":
        if uploaded_file is not None:
            image_ss = imread(image_path1)
            with st.sidebar:
                threshold_num = st.slider(label="Number of thresholds",min_value=2,max_value=25,step=1)

                # create a text area for the user to enter data
                input_text = st.text_area('Enter thresholds value')
            if (input_text != ""):
                # split the input text using "\n" as the delimiter, then convert to a list
                data_list = input_text.strip().split('\n')

            # convert the list to a NumPy array
            data_array = np.array(list(map(float, data_list)),dtype=int)

            resulted_image = spct.spectral_threshold(image, threshold_num, data_array)
            st.image(resulted_image)

    if option == "RGB to LUV":
        if uploaded_file is not None:
            image_ss = imread(image_path1)
            resulted_image = luv.rgb_to_luv_man(image_ss)
            st.image(resulted_image)

    if option == 'Region Growing':
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img = np.array(image)
            mask = RG.RegionGrowing(image_path1, (100, 100))
            segmented_image_rg =cv2.bitwise_and(img,img, mask=mask)
            segmented_image_rg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            st.image(segmented_image_rg,width=450)


#---------------------------------Thresholding------------------------------------------
if option == "Manual & Otsu's Thresholding":
    if uploaded_file is not None:
        threshold = 0; threshold_1 = 0; threshold_2 = 0; threshold_3 = 0; threshold_4 = 0
        thresh_mode = 0
        image1=cv2.imread(image_path1)
        with st.sidebar:
            edge_detect = st.selectbox("Thresholding Technique", ["Manual Thresholding", "Otsu's Thresholding"])
            if edge_detect == "Manual Thresholding":
                thresh_mode = 0
            elif edge_detect == "Otsu's Thresholding":
                thresh_mode = 1

            threshold_type = st.radio("Thresholding Type", ["Local", "Global"], horizontal=True)
            if threshold_type == "Local":
                if thresh_mode == 0:
                    threshold_1 = st.slider(label="Threshold Value 1", min_value=0, max_value=255, step=1)
                    threshold_2 = st.slider(label="Threshold Value 2", min_value=0, max_value=255, step=1)
                    threshold_3 = st.slider(label="Threshold Value 3", min_value=0, max_value=255, step=1)
                    threshold_4 = st.slider(label="Threshold Value 4", min_value=0, max_value=255, step=1)
            elif threshold_type == "Global":
                if thresh_mode == 0:
                    threshold =st.slider(label="Threshold Value", min_value=0, max_value=255, step=1)
                
        with input_img:
            image = Image.open(uploaded_file)
            # st.image(uploaded_file)
        with resulted_img:
            if threshold_type == "Local":
                # st.title("Output image")
                local_type = thresh.local_thresholding(image1, threshold_1, threshold_2, threshold_3, threshold_4, thresh_mode)
                st.image(local_type)
            elif threshold_type == "Global":
                # st.title("Output image")
                global_type = thresh.global_thresholding(image1, threshold, thresh_mode)
                st.image(global_type)
    
        