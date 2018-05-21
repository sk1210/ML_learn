import json
import pickle

dataset = "train.json"

def saveLabels(max_size = 10000):
    labels = {}
    with open(dataset, 'r') as f:
        data = json.load(f)
        print ("loaded")

        for i,image in enumerate(data["annotations"]):
            #print (image)

            imageId = int(image["imageId"])
            labelId = list(map(int,image["labelId"]))
            labels[imageId] = labelId

            if i%100 ==0 : print (i)
            #print (labels)

    f = open("labels.pkl",'wb')
    pickle.dump(labels, f)
    f.close()

    print("saved")

    #return saveLabels
#saveLabels()
labels = {}
def load_labels():
    global labels
    f = open(labels.pkl,"rb")
    labels = pickle.load(labels)
    f.close()

