
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, argparse
import spacy
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib



VQA_model_file_name      = 'models/VQA/VQA_MODEL.json'
VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name  = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights.h5'



def get_image_model(CNN_weights_file_name):
    from models.CNN.VGG import VGG_16
    image_model = VGG_16(CNN_weights_file_name)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model




def get_image_features(image_file_name, CNN_weights_file_name):
    image_features = np.zeros((1, 4096))
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    mean_pixel = [103.939, 116.779, 123.68]

    im = im.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]

    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0) 

    image_features[0,:] = get_image_model(CNN_weights_file_name).predict(im)[0]
    return image_features



def get_question_features(question):
    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, len(tokens), 300))
    for j in xrange(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


word_embeddings = spacy.load('en')


obama = word_embeddings(u"obama")
putin = word_embeddings(u"putin")
banana = word_embeddings(u"banana")
monkey = word_embeddings(u"monkey")


obama.similarity(putin)

obama.similarity(banana)

banana.similarity(monkey)


def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model


image_file_name = 'test.jpg'
question = u"What vehicle is in the picture?"

image_features = get_image_features(image_file_name, CNN_weights_file_name)

question_features = get_question_features(question)

y_output = model_vqa.predict([question_features, image_features])

labelencoder = joblib.load(label_encoder_file_name)
for label in reversed(np.argsort(y_output)[0,-5:]):
    print(str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label))



def get_image_features(image_file_name, CNN_weights_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
        
    from skimage import io
    # if you would rather not install skimage, then use cv2.VideoCapture which surprisingly can read from url
    # see this SO answer http://answers.opencv.org/question/16385/cv2imread-a-url/?answer=16389#post-id-16389
    im = cv2.resize(io.imread(image_file_name), (224, 224))
    im = im.transpose((2,0,1)) # convert the image to RGBA

    
    im = np.expand_dims(im, axis=0) 

    image_features[0,:] = get_image_model(CNN_weights_file_name).predict(im)[0]
    return image_features





image_file_name = "http://www.newarkhistory.com/indparksoccerkids.jpg"

image_features = get_image_features(image_file_name, CNN_weights_file_name)


question = u"What are they playing?"


question_features = get_question_features(question)



y_output = model_vqa.predict([question_features, image_features])

labelencoder = joblib.load(label_encoder_file_name)
for label in reversed(np.argsort(y_output)[0,-5:]):
    print(str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label))




question = u"Are they playing soccer?"


question_features = get_question_features(question)




y_output = model_vqa.predict([question_features, image_features])

labelencoder = joblib.load(label_encoder_file_name)
for label in reversed(np.argsort(y_output)[0,-5:]):
    print(str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label))


