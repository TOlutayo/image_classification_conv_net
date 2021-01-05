# ECSE 551 Mini-project 3

The accompanying notebook contains the various models that were implemented in the modified FMNIST image classification task. A guide on how to succesfully implement the models is presented. Altogether, there are 19 models that were trained and tested in this project. 
## Usage
Mount your google drive to gain access to the training data
```python
from google.colab import drive
drive.mount('/content/gdrive')
```
Read training data and labels
```python
#DATA LOADER CLASS
# Read a pickle file and disply its samples
# Note that image data are stored as unit8 so each element is an integer value between 0 and 255
data = pickle.load( open( './Train.pkl', 'rb' ), encoding='bytes')
targets = np.genfromtxt('./TrainLabels.csv', delimiter=',', skip_header=1)[:,1:]
```

## Data-Loading for Batch Processing
Before proceeding to the model selection stage, few things to consider. If you want to run the model with Data Augmentation then run block 7. If not, DO NOT RUN! Here the training set is split into 50k samples for training and 10k as hold out. 

```python
''' FOR DATA AUGMENTATION EXPERIMETNS ONLY'''
dataset = MyDataset('./Train.pkl', './TrainLabels.csv',
                transform=img_transform_DN, idx=None)
DONOTUSE, val_data = torch.utils.data.random_split(dataset, [50000,10000],generator=torch.Generator().manual_seed(42))
batch_size = 256 #feel free to change it
dataset = MyDataset('./Train.pkl', './TrainLabels.csv',transform=DA_transform, idx=None)
train_data, DONOTUSE = torch.utils.data.random_split(dataset, [50000,10000],generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

```

## Model Selection
Select the model you would like to train the data with. In this Section, only the chosen model block should be executed to avoid errors. An example of one of the models is given below
```python
class Custom2Net(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(Custom2Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7,padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 64, kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 512),
            nn.ReLU(),
            nn.Dropout2d(p=.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout2d(p=.2),
            nn.Linear(128, 9),
            nn.LogSoftmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

## Training
After executing the selected model block, the next step is to train the selected model on the sample data. In this section there's a subsection titled "Execute Model with Fourier Transformed data on ResNet18." This block should only be run if you want run the model on a fourier transformed image. This was tested with the ResNet18 Original model. However, ensure that the number of channels in the first conv laayer of the selected model has been adjusted to 3 to include magnitude and phase information obtained from the fourier transform. The training block in question is shown below. Also model parameters such as learning rate, loss function, etc. can be modified in this section.
```python
#This is used with the resnet18 model
#Change the number of channels in the first layer-
# of the model to 3 to include magnitude and phase information-
#obtained from the fourier transforms.

# Here is a piece of code that reads data in batch.
# In each epoch all samples are read in batches using dataloader
model = resnet18()
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001)

loss_fn = nn.BCEWithLogitsLoss()
num_epochs=50 # Feel free to change it

for epoch in range(num_epochs):
    for data in train_loader:
        [img,label] = data
        
        img_fft = torch.fft.fftn(img)
        img_abs = img_fft.abs()
        img_ph = img_fft.angle()
        img_abs_ph = torch.cat((img_abs,img_ph),1)
        img_tot = torch.cat((img,img_abs_ph),1)
        img_cuda = img_tot.cuda()

        output = model(img_cuda)
        label_cuda = (label-5).cuda()
        #print(output.shape)
        #_, index = torch.max(output, 1)
        #pred = output[index[0]]
        
        #index = index+8
        #print('index')
        #print(index)
        #print(index.dtype)
        
        #loss =  loss_fn(index.type(torch.double), label.type(torch.double))
        loss = F.cross_entropy(output, label_cuda)
        #loss = Variable(loss, requires_grad = True)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch', epoch, 'done')

print('Done!')


PATH = './resnet18_logsigmoid_adamW.pth'
torch.save(model.state_dict(), PATH)
```

## Validation
Just like in the training phase, the validation phase also contains two sections. The first block is for testing the model on the hold out data when not using fourier transforms. The second block in this section should only be implemented if fourier transforms were used in the previous step (training).

## Model Path Files (.pth)
Each of the 19 models has its own path file. This allows you to run the model on test data right away without having to re-train which is very time consuming. All .pth files are attached to the submission folder. There are other .pth files obtained from the aforementioned trained models but using different number of epochs and different optimizers. More on this can be found in the report. To load a model for testing, see below.
```python
#run on testing data
dataset = MyDataset('./Test.pkl', './ExampleSubmissionRandom.csv', transform=img_transform, idx=None)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


testnetwork = resnet18()
testnetwork.to(device)
PATH = './resnet18_logsigmoid_adamW.pth' #load .pth file you want to test here 
testnetwork.load_state_dict(torch.load(PATH))

count = 0
all_id = []
for data in test_loader:
    [img,label] = data
    #print('label')
    #print(label.shape)
    #print(label.dtype)
    image_cuda = img.cuda()
    output = testnetwork(image_cuda)
    _, index = torch.max(output, 1)
    pred = output[index[0]]
    index = index+5
    all_id.append(index.cpu().numpy())

    for i in range(len(label)):
      if index[i] == label[i]:
        count = count+1
    #a = sklearn.metrics.accuracy_score(label, index)
    #acc.append(a)
    
print('Done!')
```

## Results
Write predictions to a csv file with the block of code shown below.
```python
#write to csv
def write_to_csv(labs, model):
  dt = {'class': labs}
  df = pd.DataFrame(dt)
  filename = model + '_out.csv'
  df.to_csv(filename, columns=['class'],index_label=['id'])
```

```python
#write to csv

write_to_csv(np.concatenate(all_id), 'resnet18_logsigmoid')
```

## Contact
If there are any issues with running the notebok please conatct toluwaleke.olutayo@mail.mcgill.ca and I will reply immediately.

## License
[MIT](https://choosealicense.com/licenses/mit/)