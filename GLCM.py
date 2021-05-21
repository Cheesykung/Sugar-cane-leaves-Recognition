import cv2
import numpy as np
import tqdm as t
import sklearn.neighbors as sn
import skimage.feature as skf
featureTr = [];
labelTr = [];
paraQuantize = 64 
paraAngle = [0, 45, 90, 135]
paraDistance = [1, 2, 3]

# Training Image Loader and Feature Extraction
for _classname in t.tqdm(range(1,3)):
for _id in range(1,11):
path = 'C:/week10/Tr/' + str(_classname) + '/text (' + str(_id) + ').bmp';
img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
img = (img / (256/paraQuantize)).astype(int); # Image Quantization
glcm = skf.greycomatrix(img, distances=paraDistance, angles=paraAngle, 
levels=paraQuantize, symmetric=True, normed=True)
featureCon = skf.greycoprops(glcm, 'contrast')[0]
featureEne = skf.greycoprops(glcm, 'energy')[0]
featureHom = skf.greycoprops(glcm, 'homogeneity')[0]
featureCor = skf.greycoprops(glcm, 'correlation')[0]
featureTmp = np.hstack((featureCon, featureEne, featureHom, featureCor))
featureTr.append(featureTmp)
labelTr.append(_classname)
featureTr = np.array(featureTr)

# Testing Image Loader and Feature Extraction
path = 'C:/week10/Tr/2/text (9).bmp’;
img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
img = (img / (256/paraQuantize)).astype(int);
glcm = skf.greycomatrix(img, distances=paraDistance, angles=paraAngle, levels=paraQuantize, 
symmetric=True, normed=True)
featureCon = skf.greycoprops(glcm, 'contrast')[0]
featureEne = skf.greycoprops(glcm, 'energy')[0]
featureHom = skf.greycoprops(glcm, 'homogeneity')[0]
featureCor = skf.greycoprops(glcm, 'correlation')[0]
featureTs = [np.hstack((featureCon, featureEne, featureHom, featureCor))]
labelTs = 2
classifier = sn.KNeighborsClassifier(n_neighbors=1)
classifier.fit(featureTr, labelTr)
out = classifier.predict(featureTs)
print('Answer is ' + str(out))