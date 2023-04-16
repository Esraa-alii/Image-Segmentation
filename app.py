import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os
from KMean import KMeans

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
        option = st.selectbox("",["Segmentation using K-means"])
        if option == "Segmentation using K-means":
            max_iter = st.slider(label="Max number of iterations",min_value=1, max_value=100, step=2)
            k = st.slider(label="clusters",min_value=1, max_value=5, step=1)
          
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