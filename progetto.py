import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report,accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

n=10 #number of epochs
datatrain="data/Training"
datatest="data/Test"
img=(100,100)
batches=30 #number of baches of training dataset
tf.keras.utils.set_random_seed(10) #set seed value for reproducibility
    
train_dataset=tf.keras.utils.image_dataset_from_directory( #makes the computation weight lighter
    datatrain,
    label_mode='categorical',
    image_size=img,
    batch_size=batches
)

test_dataset=tf.keras.utils.image_dataset_from_directory(
    datatest,
    label_mode='categorical',
    image_size=img,
    batch_size=batches
)
model=tf.keras.Sequential([
tf.keras.layers.Rescaling(1./255,input_shape=(100,100,3)),
tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same"),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(units=64,activation='relu'),
tf.keras.layers.Dropout(0.1), 
tf.keras.layers.Dense(units=206,activation='softmax')])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

history=model.fit(
    train_dataset,
    epochs=n,
    validation_data=test_dataset
)

model.summary()

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs_labels=[]
for i in range(n):
    epochs_labels.append(i+1)

#Accuracy Graph
plt.figure(figsize=(16,6))
plt.subplot(1,2,1) 
plt.xticks(ticks=range(n),labels=epochs_labels)
plt.plot(range(n),acc,'o-',label='Training Accuracy')
plt.plot(range(n),val_acc,'o-',label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy Graph')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

#Loss Graph
plt.subplot(1,2,2)
plt.xticks(ticks=range(n),labels=epochs_labels)
plt.plot(range(n),loss,'o-',label='Training Loss')
plt.plot(range(n),val_loss,'o-',label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss Graph')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.suptitle('Performance',fontsize=16)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()
plt.close()


ytrue=[]
ypred=[]
folder="data/Test"
class_names=sorted(os.listdir(folder))

for image_batch,label_batch in test_dataset:
    ytrue.extend(np.argmax(label_batch,axis=1))
    preds=model.predict(image_batch,verbose=0)
    ypred.extend(np.argmax(preds,axis=1))

cm=confusion_matrix(ytrue,ypred,normalize='true') #confusion matrix
plt.figure(figsize=(14,12))
sns.heatmap(cm,annot=False,cmap='Blues') #set the color for heatmap

plt.title("Normalized Confusion Matrix",fontsize=16)
plt.ylabel("True Label",fontsize=12)
plt.xlabel("Predicted Label",fontsize=12)

spacing=10  #show 1/10 classes every time
positions=[] #numeric position of the labels
labels=[]  # name of classes

for i,class_name in enumerate(class_names):  #to create the notation of fruits in heat map
    if i%spacing==0:
        positions.append(i)
        labels.append(class_name)

plt.xticks(positions,labels,rotation=90)
plt.yticks(positions,labels)
plt.tight_layout()
plt.show()
plt.close()


globalacc=accuracy_score(ytrue, ypred)  # final value of val_acc of the model
print(f"Accuracy:{globalacc:.4f}") # to print 4 decimal
print(classification_report(ytrue,ypred,target_names=test_dataset.class_names)) #create the classification report

import random
def predict_image(image_path):
    image=tf.keras.utils.load_img(image_path, target_size=(100, 100))
    image_array=tf.keras.utils.img_to_array(image)
    image_batch=np.expand_dims(image_array, axis=0)
    scores=model.predict(image_batch, verbose=0)
    best_guess_index=np.argmax(scores[0])
    predicted_label=class_names[best_guess_index]
    confidence=100*np.max(scores[0])
    return predicted_label,confidence

mistakes=[]
correct=0

for actual_label in class_names:
    image_folder=os.path.join(folder,actual_label)
    random_image_file=random.choice(os.listdir(image_folder))
    path_to_image=os.path.join(image_folder,random_image_file)
    
    predicted_label,confidence=predict_image(path_to_image)
    
    if predicted_label==actual_label:
        correct+=1
    else:
        mistakes.append({
            "path":path_to_image,
            "actual":actual_label,
            "guess":predicted_label,
            "confidence":confidence
        })

total_images=len(class_names)
accuracy=(correct/total_images)*100

print(f"Accuracy: {accuracy:.2f}%")
print(f"Correct Guesses: {correct}/{total_images}")
print(f"Mistakes: {len(mistakes)}")

if mistakes:
    num_mistakes=len(mistakes)
    num_cols=4
    num_rows=(num_mistakes+num_cols-1)//num_cols
    
    plt.figure(figsize=(16,4*num_rows))
    
    for i,mistake in enumerate(mistakes):
        plt.subplot(num_rows, num_cols, i+1)
        
        image=tf.keras.utils.load_img(mistake["path"])
        plt.imshow(image)
        plt.axis('off')
        
        plt.title(
            f"Actual: {mistake['actual']}\n"
            f"Guessed: {mistake['guess']} ({mistake['confidence']:.1f}%)",
            color='red'
        )
        
    plt.tight_layout()
    plt.show()
else: 
    print("\nThe model made no mistakes in this test")

datahigh=np.loadtxt("lr_high.dat") #val_loss & val_accuracy model with eta=0.01
datalow=np.loadtxt("lr_low.dat") #val_loss & val_accuracy model with eta=0.0001
loss_low=datalow[:,2]
loss_high=datahigh[:,2]
acc_low=datalow[:,3]
acc_high=datahigh[:,3]

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.title('Validation Loss Graph')
plt.xticks(ticks=range(1,n+1)) 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1,n+1),loss_high,"o-",label="$\eta_{1}=0.01$",color="red")
plt.plot(range(1,n+1),val_loss,"o-",label="$\eta_{2}= 0.001$",color="blue")
plt.plot(range(1,n+1),loss_low,"o-",label="$\eta_{3} = 0.0001$",color="green")
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.title("Validation Accuracy Graph")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(ticks=range(1,n+1)) 
plt.plot(range(1,n+1),acc_high,"o-",label="$\eta_{1} = 0.01$",color="red")
plt.plot(range(1,n+1),val_acc,"o-",label="$\eta_{2} = 0.001$",color="blue")
plt.plot(range(1,n+1),acc_low,"o-",label="$\eta_{3} = 0.0001$",color="green")
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout(rect=[0,0,1,0.95])
plt.suptitle("Learning Rate Validation Performance")


plt.show()
plt.close()

highdrop=np.loadtxt("drop025.dat") #val_loss & val_accuracy model with dropout=0
lowdrop=np.loadtxt("drop0.dat") #val_loss & val_accuracy model with dropout=0.25
loss_low=lowdrop[:,2]
loss_high=highdrop[:,2]
acc_low=lowdrop[:,3]
acc_high=highdrop[:,3]

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.title('Validation Loss Graph')
plt.xticks(ticks=range(1,n+1)) 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1,n+1),loss_high,"o-",label="$\delta_{1}=0$",color="red")
plt.plot(range(1,n+1),val_loss,"o-",label="$\delta_{2}=0.1$",color="blue")
plt.plot(range(1,n+1),loss_low,"o-",label="$\delta_{3}=0.25$",color="green")
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.title("Validation Accuracy Graph")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(ticks=range(1,n+1)) 
plt.plot(range(1,n+1),acc_high,"o-",label="$\delta_{1}=0$",color="red")
plt.plot(range(1,n+1),val_acc,"o-",label="$\delta_{2}=0.1$",color="blue")
plt.plot(range(1,n+1),acc_low,"o-",label="$\delta_{3}=0.25$",color="green")
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout(rect=[0,0,1,0.95])
plt.suptitle("DropOut Validation Performance")