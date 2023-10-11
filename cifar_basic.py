import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from basic_net import Net
import torch.optim as optim

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

training_file = unpickle('data_batch_1')

training_data = training_file[b'data']
training_labels = training_file[b'labels']

train_data_transformed = []
for image in training_data:
    #32 pixels by 32 pixels by R, G, and B
    new_image = np.zeros((32,32,3)).astype('uint8')
    for i in range(32):
        for j in range(32):
            r = image[i * 32 + j]
            g = image[i * 32 + j + 1024]
            b = image[i * 32 + j + 2048]
            #convert 1D array to 3D
            new_image[i][j][0] = r
            new_image[i][j][1] = g
            new_image[i][j][2] = b
    train_data_transformed.append(new_image)
#Now the new images are stored 

classes = unpickle('batches.meta')[b'label_names']

'''
plt.imshow(train_data_transformed[4200])
plt.title(classes[training_labels[4200]])
plt.show()
'''

#shape = (num images, rgb, i, j)
train_data_transformed = np.array(train_data_transformed)
train_data_transformed = np.rollaxis(train_data_transformed, 3, 1)
train_data_transformed = train_data_transformed / 255.0

training_data = torch.from_numpy(train_data_transformed).to(torch.float32)
def get_one_hot(target, nb_classes):
    arr = np.eye(nb_classes)[np.array(target).reshape(-1)]
    return arr.reshape(list(target.shape) + [nb_classes])

training_labels = np.array(training_labels)
training_labels.shape
#training_labels = get_one_hot(training_labels, 10)


training_labels = torch.tensor(training_labels, dtype=torch.long)
print(training_labels.shape)
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = .001, momentum = 0.9)

for epoch in range(50):
    running_loss = 0
    correct = 0
    total = 0
    for i, data in enumerate(training_data):
        inputs = data.unsqueeze(0)
        labels = training_labels[i].unsqueeze(0)

        optimizer.zero_grad()
        
        outputs = net(inputs)

        #compare output to the label
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 1000 == 999:
            print(f'[epoch: {epoch + 1}, iteration: {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0
            
    epoch_accuracy = 100 * (correct / total)
    print(epoch_accuracy)