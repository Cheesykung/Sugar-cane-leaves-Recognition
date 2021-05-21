import cv2
import numpy as np
import tqdm as t
import sklearn.neighbors as sn
featureTr = [];
labelTr = [];
for _classname in t.tqdm(range(1,3)):
for _id in range(1,8):
path = 'C:/Cal/Tr/' + str(_classname) + '/img (' + str(_id) + ').jpg';
img = cv2.imread(path)
# แปลงภำพให้อยู่บนปริภูมิสีHSV
out = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# แปลงข้อมูลจำกเมตริกซ์ให้อยู่ในรูปแบบเวกเตอร์ เฉพำะ Hue
out = out[:,:,0].reshape(1,-1);
# สร้ำงฮิสโตแกรมจำก Hue
hist, bins = np.histogram(out,bins = np.arange(-0.5,256,1) )
# Normalization เพื่อท ำให้ Feature รสำมำรถรองรับขนำดภำพที่แตกต่ำงกันได้
featureTr.append([hist/np.sum(hist)])
labelTr.append(_classname)

featureTr = np.array(featureTr)
featureTr = np.reshape(featureTr,(14,256))
path = 'C:/Cal/Tr/2/img (5).jpg’;
img = cv2.imread(path)
out = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
out = out[:,:,0].reshape(1,-1);
hist, bins = np.histogram(out,bins = np.arange(-0.5,256,1) ) 
tmp = [hist/np.sum(hist)]
featureTs = (np.array([tmp])).reshape(1,-1)
labelTs = 2
classifier = sn.KNeighborsClassifier(n_neighbors=1)
classifier.fit(featureTr, labelTr)
out = classifier.predict(featureTs)
print('Answer is ' + str(out))