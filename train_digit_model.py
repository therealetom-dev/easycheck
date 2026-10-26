"""Offline-only training of the small secondary digit classifier; not run in deployment.
Trains on scikit-learn 8x8 handwritten digits dataset (UCI Optical Recognition
of Handwritten Digits). Inference uses cv2 only. Does not imply camera-photo accuracy.
"""
from pathlib import Path
import cv2
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT=Path(__file__).resolve().parent
data=load_digits()
features=data.data.astype(np.float32)
labels=data.target.astype(np.int32)
Xtrain,Xtest,ytrain,ytest=train_test_split(features,labels,stratify=labels,random_state=42,test_size=.25)
svm=cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(8)
svm.setGamma(0.0005)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,10000,1e-6))
svm.train(Xtrain,cv2.ml.ROW_SAMPLE,ytrain)
_,predict=svm.predict(Xtest)
print('Held-out *dataset* accuracy (not camera accuracy):',round(accuracy_score(ytest,predict.ravel()),4))
out=ROOT/'assets'/'digits_svm.xml'
svm.save(str(out))
print('Model',out,round(out.stat().st_size/1024),'KB')
