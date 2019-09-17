import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
#taking the input and pre processing it

def get_inputs(name):    
    global x_train
    x_train = np.genfromtxt(name+'_x_train.csv',delimiter=',')  
    x_train = x_train.astype(int)
    global y_train
    y_train = np.genfromtxt(name+'_y_train.csv' ,delimiter=',')  
    y_train = y_train.astype(int)
    global  x_test
    x_test = np.genfromtxt(name+'_x_test.csv',delimiter=',')
    global y_test
    y_test = np.genfromtxt(name+'_y_test.csv',delimiter=',')
    
#  Creating a model and training it
    
def build_model(EPOCH_COUNT,B_SIZE):
    global model
    global history
    model = Sequential()
    model.add(Dense(32, input_dim=10, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    history = model.fit(x_train , y_train,
              epochs=EPOCH_COUNT,
              batch_size=B_SIZE)
#    score = model.evaluate(x_train, y_train, batch_size=1)
    print(model.summary())
#creating a file containing no of weights and also the weight and bias matrix in csv format

def save_weights(name):
    wt = model.get_weights()
    w=0
    b=0
    for count in wt:    #(weight/bias)_layer number_(rowxcol)
        if(len(count.shape)>1):
            w+=1
        else:
            b+=1
    l = [w]
        
    np.savetxt(name + '_numbers_layer.csv', l)
    
    layer_number = 0
    for wg in wt:    #(weight/bias)_layer number_(rowxcol)
        
#        row = wg.shape[0]
        if(len(wg.shape)>1):
#            col = wg.shape[1]
            rxc = (name)+'_wt_'+str(int(layer_number/2))+'.csv'
            
        else:
            rxc = (name)+'_bi_'+str(int(layer_number/2))+'.csv'
            
        np.savetxt(rxc, wg, delimiter=",")
        layer_number+=1
    return wt

#predicting the output for the ith no of input and storing it in output.csv format

def evaluate(name, ith_row):
    x_in = x_train[ith_row]
    q=[]
    count_element = 0
    for kth_element in list(x_in):
        
        l=[]
        q.append(l)
        q[count_element].append(kth_element)
        count_element+=1
    r = np.array(q)
    r = np.transpose(r)
    output= model.predict(r)
    np.savetxt(name +'_output.csv', output, delimiter=",")
    return output

def evaluate_all(name):
    total = []
    for i_row in range(2000):
        n = evaluate(name,i_row)
        n.astype(int)
    
        total.append(n[0])
    np.savetxt(name+'_output.csv', total, delimiter=",")
    
def plot(name):
    #y_train1 = np.genfromtxt(name+'_y_train.csv',delimiter=',')  
    k_output = np.genfromtxt(name +'_output.csv',delimiter=',')  
    diff = y_train-k_output
    np.savetxt(name +'_diff.csv', diff, delimiter=",")
    plt.plot(diff/max(name+'_y_train'))
    plt.title(' Error Plot Min:'+ (min(diff)).astype(str)+' Max:'+ (max(diff)).astype(str))
    plt.show()
    # Plot training & validation accuracy values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return diff

#################################################################################
#the main code for user
#enter name in string format
file_name = 'netcube'

get_inputs(file_name)
build_model(3000,32)#nter epoch_count for training
save_weights(file_name)
evaluate_all(file_name)
a=plot(file_name)
