import torch
import torch.nn as nn
import torch.optim as opt
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
class_names=['airplane', 'automobile', 'bird' , 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

device="cuda" if torch.cuda.is_available() else "cpu"
print("using device " ,device)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data=CIFAR10(root="data",train=True,download=True,transform=train_transform)
test_data=CIFAR10(root="data",train=False,download=True,transform=test_transform)
train_loader=DataLoader(train_data, batch_size=64,shuffle=True )
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)


class cifar_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2,2))
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,2))
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2,2))
        self.conv4=nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.classifier=nn.Sequential(nn.Flatten(),
                                      nn.Linear(256*4*4,512),nn.ReLU(),nn.Dropout(p=0.5),
                                      nn.Linear(512,256),nn.ReLU(),nn.Dropout(p=0.3),
                                      nn.Linear(256,10))
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.classifier(x)
        return x
    
loss_fn=nn.CrossEntropyLoss()
model=cifar_module().to(device)
optimizer=opt.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
scheduler=opt.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
epochs=30
for epoch in range(epochs):
    model.train()
    total_loss=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        output=model(images)
        loss=loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    scheduler.step()
    print(f"epoch  :  {epoch+1}/{epochs}    |  avg loss: {total_loss / len(train_loader):.4f}")

model.eval()
correct=0
total=0
with torch.no_grad():
        for images,labels in test_loader:
         images=images.to(device)
         labels=labels.to(device)
         outputs=model(images)
         predictions=outputs.argmax(dim=1)
         correct+=(predictions==labels).sum().item()
         total+=labels.size(0)
accuracy=100*correct/total
print(f"accuracy : {accuracy:.2f} %")
torch.save(model.state_dict(),"cifar_10_model.pth")
print("model saved as  : cifar_10_model.pth")
index=int(input("enter the index of the CIFAR-10 test image (0-9999): "))
image,true_label=test_data[index]
img_display = image.permute(1, 2, 0) * 0.5 + 0.5 
plt.imshow(img_display)  
plt.title(f"actual label : {class_names[true_label]}")
plt.axis("off")
plt.show()
image_flat=image.unsqueeze(0).to(device)
with torch.no_grad():
    output=model(image_flat)
    predicted_label=output.argmax(dim=1).item()
print("user picked image index", index)
print("actual label" ,class_names[true_label])
print("model prediction ", class_names[predicted_label])