#Evolutionary Machine Learning
#Home Work 1
#Author: Suket Singh | SuID: 923277656


import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from geneticalgs import BinaryGA, RealGA, DiffusionGA, MigrationGA
from sklearn.metrics import accuracy_score

# apply seed for reproducing same results
seed = 9
np.random.seed(seed)

# load cryotherapy.csv data set
dataset = np.loadtxt('Cryotherapy.csv', delimiter=',', skiprows=1)

# split into input and output variables
X = dataset[:,0:6]
Y = dataset[:,6]

# split the data into training (80%) and testing (20%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.20, random_state=seed)

# create the model
model = Sequential()
model.add(Dense(6, input_dim=6, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=180, batch_size=5)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

# REFERENCE: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

plt.plot(history.history['loss'])

plt.title("Back PropogationAccuracy: %.2f%%" %(scores[1]*100))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Settings for GA
generation_num = 50
population_size = 16
elitism = True
selection = 'rank'
tournament_size = None # for tournament selection
mut_type = 1
mut_prob = 0.1
cross_type = 1
cross_prob = 0.95
optim = 'min' # minimize or maximize a fitness value? May be 'min' or 'max'.
interval = (-1, 1)



# Migration settings for GA
period = 5
migrant_num = 3
cloning = True

def rand(x):
    return abs(x*(math.sin(x/11)/5 + math.sin(x/110)))

x1 = list(range(1000))
y1 = [rand(elem) for elem in x1]

a = np.array(y1[0:36]).reshape(6,6)
b = np.array(y1[36:42])
c = np.array(y1[42:48]).reshape(6,1)
d = np.array(y1[54:55])

model.layers[0].set_weights([a,b])
model.layers[1].set_weights([c,d])

weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]
print(weights, biases)

from sklearn.metrics import confusion_matrix

def err_funct(x):

    a = np.array(x[0:36]).reshape(6,6)
    b = np.array(x[36:42])
    c = np.array(x[42:48]).reshape(6,1)
    d = np.array(x[54:55])
   
    
    model.layers[0].set_weights([a,b])
    model.layers[1].set_weights([c,d])
    
    y_predict = model.predict(X_train)
    
#    i = 0
#    while i < len(y_predict):
#        if y_predict[i][0] < 0.5:
#            y_predict[i][0] = 0
#        else:
#            y_predict[i][0] = 1
#        i += 1    

    # Calculate the RMSE score when opting to minimize fitness in GA
    #rmse = np.sqrt(mean_squared_error(Y_train, y_predict))
    y_predict=y_predict.flatten()
    rmse = np.sum(np.square(y_predict - Y_train))
    #print('Validation RMSE: ', rmse,' ')
    
        
    # Calculate the accuracy score when opting to maximize fitness in GA
    #acc = accuracy_score(Y_train, y_predict)
    #print('Accuracy Score: ', acc, '\n')
    
    return rmse

sga = RealGA(err_funct, optim=optim, elitism=elitism, selection=selection,
            mut_type=mut_type, mut_prob=mut_prob, 
            cross_type=cross_type, cross_prob=cross_prob)

sga.init_random_population(population_size, 97, interval)

fitness_progress = sga.run(generation_num)

    #return np.sum(np.square(y_predict - Y_train))

sga.best_solution

best_weights = sga.best_solution[0]

a = np.array(best_weights[0:36]).reshape(6,6)
b = np.array(best_weights[36:42])
c = np.array(best_weights[42:48]).reshape(6,1)
d = np.array(best_weights[54:55])

model.layers[0].set_weights([a,b])
model.layers[1].set_weights([c,d])

Y_test_p = model.predict(X_test)

print(Y_test)

print(Y_test_p)


i = 0
while i < len(Y_test_p):
    if Y_test_p[i][0] < .405:
        Y_test_p[i][0] = 0
    else:
        Y_test_p[i][0] = 1
    i += 1 
        
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_test_p)
print(cm)

acc = accuracy_score(Y_test, Y_test_p)
print('Accuracy Score: ', acc, '\n')


