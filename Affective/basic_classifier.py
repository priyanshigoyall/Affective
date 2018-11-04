import random
import pprint
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import tree
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

def cleanData(text):
    
    stop_words = set(stopwords.words('english'))
    to_remove = ".)(:,!?\n"
    table = {ord(char): None for char in to_remove}
    w = []
    words = text.split(' ')
    for word in words:
        word = word.translate(table).lower()
        if word not in stop_words:
            w.append(word)
    text = " ".join(w)
    return text


def importTraining():
    train_data = []
    train_class_emotion = []
    train_class_anger_intensity = []
    train_class_anger = []
    train_class_sadness_intensity  = []
    train_class_sadness = []
    train_class_fear = []
    train_class_fear_intensity = []
    train_class_joy = []
    train_class_joy_intensity = []
    file = open('anger-ratings-0to1.train.txt', encoding = 'utf-8')
    for lines in file:
        line = lines.split('\t')
        clean_text = cleanData(line[1])
        intensity = 0.0
        try:
            intensity = float(line[3].strip('\n'))
        except ValueError:
            #print('Line {i} is corrupt!')
            intensity = 0.0
        train_class_anger.append(clean_text.encode())   
        train_data.append(clean_text.encode())
        train_class_emotion.append(0)
        train_class_anger_intensity.append(intensity)
    file.close()

    file = open('fear-ratings-0to1.train.txt', encoding = 'utf-8')
    for lines in file:
        line = lines.split('\t')
        clean_text = cleanData(line[1])
        intensity = 0.0
        try:
            intensity = float(line[3].strip('\n'))
        except ValueError:
            print('Line {i} is corrupt!')
            intensity = 0.0
        train_class_fear.append(clean_text.encode())
        train_data.append(clean_text.encode())
        train_class_emotion.append(1)
        train_class_fear_intensity.append(intensity)
    file.close()

    file = open('sadness-ratings-0to1.train.txt', encoding = 'utf-8')
    for lines in file:
        line = lines.split('\t')
        clean_text = cleanData(line[1])
        intensity = 0.0
        try:
            intensity = float(line[3].strip('\n'))
        except ValueError:
            print('Line {i} is corrupt!')
            intensity = 0.0
        train_class_sadness.append(clean_text.encode())   
        train_data.append(clean_text.encode())
        train_class_emotion.append(2)
        train_class_sadness_intensity.append(intensity)
    file.close()

    file = open('joy-ratings-0to1.train.txt', encoding = 'utf-8')
    for lines in file:
        line = lines.split('\t')
        clean_text = cleanData(line[1])
        intensity = 0.0
        try:
            intensity = float(line[3].strip('\n'))
        except ValueError:
            print('Line {i} is corrupt!')
            intensity = 0.0
        train_class_sadness.append(clean_text.encode()) 
        train_data.append(clean_text.encode())
        train_class_emotion.append(3)
        train_class_joy_intensity.append(intensity)
    file.close()
    
    return train_data, train_class_emotion, train_class_fear, train_class_fear_intensity, train_class_anger, train_class_anger_intensity, train_class_joy, train_class_joy_intensity, train_class_sadness,train_class_sadness_intensity

def importTesting():
    test_data = []
    test_class_emotion = []
    test_class_anger_intensity = []
    test_class_anger = []

    test_class_fear = []
    test_class_fear_intensity = []
    
    test_class_sadness = []
    test_class_sadness_intensity = []

    test_class_joy = []
    test_class_joy_intensity = []


    file = open('anger-ratings-0to1.dev.gold.txt', encoding = 'utf-8')
    for lines in file:
        line = lines.split('\t')
        clean_text = cleanData(line[1])
        intensity = 0.0
        try:
            intensity = float(line[3].strip('\n'))
        except ValueError:
            print('Line {i} is corrupt!')
            intensity = 0.0
        test_class_anger.append(clean_text.encode())   
        test_data.append(clean_text.encode())
        test_class_emotion.append(0)
        
        test_class_anger_intensity.append(intensity)
    file.close()

    file = open('fear-ratings-0to1.dev.gold.txt', encoding = 'utf-8')
    for lines in file:
        line = lines.split('\t')
        clean_text = cleanData(line[1])
        intensity = 0.0
        try:
            intensity = float(line[3].strip('\n'))
        except ValueError:
            print('Line {i} is corrupt!')
            intensity = 0.0
        test_class_fear.append(clean_text.encode())   
        test_data.append(clean_text.encode())
        test_class_emotion.append(1)
        
        test_class_fear_intensity.append(intensity)
    file.close()

    file = open('joy-ratings-0to1.dev.gold.txt', encoding = 'utf-8')
    for lines in file:
        line = lines.split('\t')
        clean_text = cleanData(line[1])
        intensity = 0.0
        try:
            intensity = float(line[3].strip('\n'))
        except ValueError:
            print('Line {i} is corrupt!')
            intensity = 0.0
            
        test_data.append(clean_text.encode())
        test_class_emotion.append(3)
        test_class_joy.append(clean_text.encode())
        test_class_joy_intensity.append(intensity)
    file.close()

    file = open('sadness-ratings-0to1.dev.gold.txt', encoding = 'utf-8')
    for lines in file:
        line = lines.split('\t')
        clean_text = cleanData(line[1])
        intensity = 0.0
        try:
            
            intensity = float(line[3].strip('\n'))
        except ValueError:
            print('Line {i} is corrupt!')
            intensity = 0.0
            
        test_data.append(clean_text.encode())
        test_class_emotion.append(2)
        test_class_sadness.append(intensity)
        #print(intensity)
        test_class_sadness_intensity.append(intensity)
    file.close()
    
    return test_data, test_class_emotion, test_class_fear, test_class_fear_intensity, test_class_anger, test_class_anger_intensity, test_class_joy, test_class_joy_intensity, test_class_sadness,test_class_sadness_intensity


def main():
    #print(float('0.229\n'.strip('\n')))
    train_data, train_class_emotion, train_class_fear, train_class_fear_intensity, train_class_anger, train_class_anger_intensity, train_class_joy, train_class_joy_intensity, train_class_sadness,train_class_sadness_intensity = importTraining()
    test_data, test_class_emotion, test_class_fear, test_class_fear_intensity, test_class_anger, test_class_anger_intensity, test_class_joy, test_class_joy_intensity, test_class_sadness,test_class_sadness_intensity = importTesting()


    #print(train_data)
    #print(train_class_intensity)

    pipeline_tfidf_logistic_regression = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1,2), max_features = 100000)), ('classifier', LogisticRegression())])
    pipeline_tfidf_decision_systems= Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1,2), max_features = 100000)), ('classifier', tree.DecisionTreeClassifier())])
    pipeline_tfidf_multilayer_perceptron = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1,1), max_features=100000)), ('classifier', MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1))])
    pipeline_tfidf_perceptron = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1,1), max_features=100000)), ('classifier', Perceptron())])

    pipeline_tfidf_linear_regression = Pipeline(steps = [('vectorizer', TfidfVectorizer(ngram_range = (1,1), max_features=100000)), ('classifier', linear_model.LinearRegression())]) 
    
    pipeline_tfidf_logistic_regression.fit(train_data, train_class_emotion)
    pipeline_tfidf_decision_systems.fit(train_data, train_class_emotion)
    pipeline_tfidf_multilayer_perceptron.fit(train_data, train_class_emotion)
    pipeline_tfidf_perceptron.fit(train_data, train_class_emotion)

    #pipeline_tfidf_linear_regression.fit(train_data, train_class_intensity)
    predict = pipeline_tfidf_logistic_regression.predict(test_data)
    predicted_class_fear = []
    predict_class_fear_intensity = []
    print(predict)
    ''' for prediction in predict:
        
        if prediction == 0:
            predicted_class_anger
        if prediction == 1:
            pipeline_tfidf_linear_regression.fit(train_class_fear, train_class_fear_intensity)
            intensity = pipeline_tfidf_linear_regression.predict(row)
        if prediction == 2:
            pipeline_tfidf_linear_regression.fit(train_class_sadness, train_class_sadness_intensity)
            intensity = pipeline_tfidf_linear_regression.predict(row)
        if prediction == 3:
            pipeline_tfidf_linear_regression.fit(train_class_joy, train_class_joy_intensity)
            intensity = pipeline_tfidf_linear_regression.predict(row)
        print(prediction, intensity) '''

    print("Logistic Regression:",pipeline_tfidf_logistic_regression.score(test_data, test_class_emotion))
    print("Decision Systems:", pipeline_tfidf_decision_systems.score(test_data, test_class_emotion))
    print("Multilayer Perceptron:", pipeline_tfidf_multilayer_perceptron.score(test_data, test_class_emotion))
    print("Perceptron:", pipeline_tfidf_multilayer_perceptron.score(test_data, test_class_emotion))

    #print('Variance score: {}'.format(pipeline_tfidf_linear_regression.score(test_data, test_class_intensity)))


main()
