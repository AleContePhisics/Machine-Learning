import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import json


n=10 #number of epochs
datatrain="data/Training"
datatest="data/Test"
img=(100,100)
batchsize=30


namefile="fruitsmodel.keras"
model=None
historydata=None
filehistory="fruitshistory.json" 

if os.path.exists(namefile):
    print(f"Model '{namefile}' found.")
    model = tf.keras.models.load_model(namefile)
    if os.path.exists(filehistory):
        print(f"File '{filehistory}' found.")
        with open(filehistory, 'r') as f:
            historydata = json.load(f)
    else:
        print("File not found")

else:
    print("Loading new Model")

    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        datatrain,
        label_mode='categorical',
        image_size=img,
        batch_size=batchsize
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        datatest,
        label_mode='categorical',
        image_size=img,
        batch_size=batchsize
    )

    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255,input_shape=(100,100,3)),
    tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(0.1), 
    tf.keras.layers.Dense(units=206, activation='softmax')
    ])
    
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    
    history=model.fit(
        train_dataset,
        epochs=n,
        validation_data=test_dataset
    )
    
    print(f"\nNew model saved in '{namefile}'")
    model.save(namefile)
    print(f"\nHistory saved in '{filehistory}'")
    # history converts in json file
    with open(filehistory, 'w') as f:
        json.dump(history.history, f)
    
    historydata = history.history

model.summary()

if historydata is not None:    
    acc=historydata['accuracy']
    val_acc=historydata['val_accuracy']
    loss=historydata['loss']
    val_loss=historydata['val_loss']
    epochs_range=range(len(acc))
    epochs_labels=[]
    for i in epochs_range:
        epochs_labels.append(i+1)


    #Accuracy Graph
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1) 
    plt.xticks(ticks=epochs_range, labels=epochs_labels)
    plt.plot(epochs_range,acc,'o-',label='Training Accuracy')
    plt.plot(epochs_range,val_acc,'o-',label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    #Loss Graph
    plt.subplot(1,2,2)
    plt.xticks(ticks=epochs_range, labels=epochs_labels)
    plt.plot(epochs_range,loss,'o-',label='Training Loss')
    plt.plot(epochs_range,val_loss,'o-',label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.suptitle('Performance', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    
else:
    print("\nThere's no cronology .")

def predictimage(image_path):
    image=tf.keras.utils.load_img(image_path, target_size=(100, 100))
    image_array=tf.keras.utils.img_to_array(image)
    image_batch=np.expand_dims(image_array, axis=0)
    predictions=model.predict(image_batch, verbose=0)
    top_prediction_index=np.argmax(predictions[0])
    predicted_label=class_names[top_prediction_index]
    confidence_score=100*np.max(predictions[0])
    return predicted_label,confidence_score

directory="data/Test"
class_names=sorted(os.listdir(directory))
errors_found=[]
correct_predictions=0

for true_label in class_names:
    class_folder=os.path.join(directory, true_label)
    random_image=random.choice(os.listdir(class_folder))
    image_to_test_path=os.path.join(class_folder, random_image)
    
    predicted_label,confidence=predictimage(image_to_test_path)
    
    if predicted_label==true_label:
        correct_predictions+=1
    else:
        errors_found.append({
            "path":image_to_test_path,
            "true_label":true_label,
            "predicted_label":predicted_label,
            "confidence":confidence
        })

total_tests=len(class_names)
accuracy=(correct_predictions/total_tests)*100
print("\nTest Results:")
print(f"Accuracy:{accuracy:.2f}%")
print(f"Correct Predictions:{correct_predictions}/{total_tests}")
print(f"Errors:{len(errors_found)}")

if errors_found:    
    num_errors=len(errors_found)
    num_cols=4
    num_rows=(num_errors+num_cols-1)//num_cols
    
    plt.figure(figsize=(16,4*num_rows))
    
    for i,error in enumerate(errors_found):
        plt.subplot(num_rows, num_cols, i + 1)
        image = tf.keras.utils.load_img(error["path"])
        plt.imshow(image)
        plt.axis('off')
        
        plt.title(
            f"True:{error['true_label']}\n"
            f"Predicted:{error['predicted_label']}\n"
            f"Confidence:{error['confidence']:.1f}%", 
            color='red'
        )
        
    plt.tight_layout()
    plt.show()
else: 
    print("No Error Found")

from sklearn.metrics import classification_report

ytrue = []
ypred = []

for image_batch, label_batch in test_dataset:
    ytrue.extend(np.argmax(label_batch, axis=1))
    preds = model.predict(image_batch, verbose=0)
    ypred.extend(np.argmax(preds, axis=1))

print("\nFinal Report:")
print(classification_report(ytrue,ypred,target_names=class_names,zero_division=0))
