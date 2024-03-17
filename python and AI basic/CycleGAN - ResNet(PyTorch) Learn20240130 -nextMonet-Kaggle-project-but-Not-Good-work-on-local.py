"""
CycleGAN
CycleGANs learns to transform images from one sample into images that could seems reasonable similar to another sample.
For example, a CycleGAN produced the right hand image below when given the left hand image as input. 
It even took an image of a horse and turned it into an image of a zebra. 
Our task is to produce images in the style of Monet using a random set of photos.
"""

#key word
#Generative Adversarial Network
#a generator and a discriminator
#fake (label 0)

import os                   # os ,file
import cv2                  #open cv images ,videos
import torch                #Pytorch Tensor,Neural network,GPU
import torch.nn as nn       #Neural Network,define Layer,loss functions,optimization methods
import numpy as np          #numerial computing,MultiDimensional Array,Vectorized Opertions
from PIL import Image       #image Lib
import shutil               #shell utilities
import itertools            #iteration
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt                     #MATLIB-like ,visualizations,plots,histograms
import torchvision.transforms as transforms
from torchvision.utils import make_grid

BATCH_SIZE = 5
MONET_IMAGES_PATH = "/kaggle/input/gan-getting-started/monet_jpg"
TEST_IMAGES_PATH = "/kaggle/input/gan-getting-started/photo_jpg"

lr = 0.0001
beta1 = 0.5
beta2 = 0.996
n_epoches = 1 #90
decay_epoch = 40
display_epoch = 25

#Dataset Transform ---------------------------------------------
"""
the transforms_dataset object created by composing these transformations together can be applied to an image dataset 
to perform random horizontal flipping, 
convert the images to tensors, and normalize them for training or evaluation of a deep learning model.

transforms.Compose([...]): This function composes several transformations together. 
It takes a list of transformation objects as input and returns a callable object that applies each transformation in the list, 
in sequential order, to the input data.

transforms.RandomHorizontalFlip(): This transformation randomly flips the input image horizontally with a probability of 0.5. 
Horizontal flipping is a common data augmentation technique used to increase the variability of the training dataset.
 It helps the model generalize better by exposing it to slightly different versions of the same image.

transforms.ToTensor(): This transformation converts the input image
 (which could be in various formats like PIL Image or numpy.ndarray) into a PyTorch tensor.
   PyTorch tensors are the primary data structure used for storing and manipulating data in PyTorch.
     This transformation also scales the pixel values from the range [0, 255] to the range [0.0, 1.0].

transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)): 
This transformation normalizes the tensor by subtracting the mean and dividing by the standard deviation along each channel of the image.
 The first tuple (0.5, 0.5, 0.5) represents the mean values for each channel (red, green, blue), 
 and the second tuple (0.5, 0.5, 0.5) represents the standard deviation for each channel. 
 This particular normalization brings the pixel values to the range [-1.0, 1.0], 
 which is a common practice in deep learning models as it helps in stabilizing the training process.
"""
transforms_dataset = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

"""
This code defines a custom dataset class named ImageDataset, which is a subclass of PyTorch's Dataset class. This class is designed to load and preprocess image data for use in deep learning models, particularly for tasks like image-to-image translation or image style transfer.

Let's break down the code:

__init__(self, directory_x, directory_y, test=False, transforms=None): This method serves as the constructor for the ImageDataset class. It initializes the dataset object with the directories containing the image data (directory_x and directory_y), a flag test indicating whether the dataset is for testing, and an optional parameter transforms for preprocessing transformations. Inside this method, the dataset is split into two parts: monet_images_X and test_images_Y, depending on whether it's a test dataset or not.

__len__(self): This method returns the total number of samples in the dataset. In this case, it returns the length of the monet_images_X list, which represents the number of samples in the dataset.

__getitem__(self, index): This method is responsible for loading and preprocessing a single sample from the dataset. It takes an index as input and returns the corresponding pair of images (x_img and y_img). Inside this method, it loads the images located at the specified indices from the directories monet_images_X and test_images_Y. If preprocessing transformations are specified (transforms is not None), it applies these transformations to the images before returning them.

Overall, this class provides an interface to load paired images from two directories (e.g., input and target images) and allows for applying preprocessing transformations. It is designed to be used in conjunction with PyTorch's DataLoader class for efficient data loading during training or evaluation of deep learning models.
"""

"""
!!! !!! !!! >>> >>> >>> 
The double underscores "__" in Python are typically pronounced as "dunder."
!!! !!! !!! >>> >>> >>> 

 "Dunder" is a portmanteau of "double" and "underscore." So, when referring to "init" in Python, you would pronounce it as "dunder init" or "double underscore init." Similarly, "len" would be pronounced as "dunder len" or "double underscore len," and "getitem" would be "dunder getitem" or "double underscore getitem." This pronunciation convention is commonly used among Python developers to refer to special methods or attributes with double underscores.
"""
"""
In Python, methods with names surrounded by double underscores like __init__, __len__, and __getitem__ are called "dunder methods" or "magic methods". These methods have special meanings in Python classes and are called automatically under certain circumstances.

__init__: This method is called when an instance of the class is created. It initializes the object's state. In this specific class ImageDataset, __init__ is used to set up the dataset, specifying directories for image loading and any transformations to be applied.

__len__: This method returns the length of the object when the built-in len() function is called on it. In this context, __len__ is used to determine the total number of samples in the dataset.

__getitem__: This method allows instances of the class to be accessed using the indexing syntax ([]). It defines the behavior for indexing operations like dataset[index]. In this class, __getitem__ is used to retrieve a specific sample (pair of images) from the dataset.

Using dunder methods like __init__, __len__, and __getitem__ allows the class ImageDataset to integrate seamlessly with Python's built-in functions and syntax, making it easier to work with instances of the class in a manner consistent with other Python objects. This is especially useful when using the class with other Python constructs or libraries that rely on these methods, such as PyTorch's DataLoader which utilizes __len__ and __getitem__ to iterate over the dataset.
"""
class ImageDataset(Dataset):
    def __init__(self, directory_x, directory_y, test=False, transforms=None):
        self.transforms = transforms
        
        if test:
            self.monet_images_X = [directory_x + "/" + name for name in sorted(os.listdir(directory_x))[250:]]
            self.test_images_Y = [directory_y + "/" + name for name in sorted(os.listdir(directory_y))[250:301]]
        else:
            self.monet_images_X = [directory_x + "/" + name for name in sorted(os.listdir(directory_x))[:250]]
            self.test_images_Y = [directory_y + "/" + name for name in sorted(os.listdir(directory_y))[:250]]
        
    def __len__(self):
        return len(self.monet_images_X)
    
    def __getitem__(self, index):
        x_img =  Image.open(self.monet_images_X[index])
        y_img =  Image.open(self.test_images_Y[index])
        
        if self.transforms is not None:
            x_img = self.transforms(x_img)
            y_img = self.transforms(y_img)
        return x_img, y_img
    
"""
DataLoader: DataLoader is a PyTorch class used for loading data in batches during the training or evaluation of a deep learning model. 
It provides various features like batching, shuffling, and parallel data loading using multiple worker processes.
batch_size = BATCH_SIZE: This parameter specifies the number of samples in each batch. 
It is a hyperparameter that controls the size of the mini-batch used during training. 
The variable BATCH_SIZE should be defined elsewhere in the code, representing the desired batch size.

shuffle = True: This parameter determines whether the data will be shuffled before each epoch. 
Shuffling the data helps prevent the model from memorizing the order of the training examples and promotes better generalization.

num_workers = 3: This parameter specifies the number of worker processes to use for data loading. 
It determines how many subprocesses will be used for data loading. Using multiple workers can speed up data loading, 
especially if loading the data involves disk I/O operations.

So, the train_loader object created by the DataLoader class will load the training data in batches, 
shuffle it before each epoch, and use three worker processes for parallel data loading. 
This loader will be used during the training process of a deep learning model.

"""
train_loader = DataLoader(
    ImageDataset(directory_x=MONET_IMAGES_PATH, directory_y=TEST_IMAGES_PATH, test=False, transforms=transforms_dataset),
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 3
)

test_loader = DataLoader(
    ImageDataset(directory_x=MONET_IMAGES_PATH, directory_y=TEST_IMAGES_PATH, test=True, transforms=transforms_dataset),
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 3
)

# CycleGAN Discriminator ------------------------------------------
"""
This code defines a neural network model for image discrimination, commonly used in tasks like image classification or image generation. Here's the breakdown of the code:

Class Definition:

class Discriminator(nn.Module): : Defines a class named Discriminator that inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
Initialization Method (__init__):

def __init__(self, in_channels): : Constructor method for the Discriminator class. It takes one parameter in_channels, which specifies the number of input channels for the first convolutional layer.
super(Discriminator, self).__init__(): Calls the constructor of the parent class nn.Module to initialize the Discriminator class.
self.scale_factor = 16: Sets a scale factor attribute, though it's not used in the provided code snippet.
self.model = nn.Sequential(...) : Defines the architecture of the discriminator using a sequential container nn.Sequential, which sequentially stacks layers.
Model Architecture (nn.Sequential):

The model consists of a series of convolutional layers followed by leaky ReLU activation functions and instance normalization layers.
nn.Conv2d: 2D convolutional layer.
nn.LeakyReLU: Leaky Rectified Linear Unit activation function, introducing a small gradient when the unit is not active.
nn.InstanceNorm2d: Instance normalization layer, which normalizes each channel of the input separately for each sample in a batch.
nn.ZeroPad2d: Zero-padding layer, used to pad the input tensor symmetrically.
The architecture gradually increases the number of channels while decreasing the spatial dimensions of the feature maps through successive convolutional layers with stride 2.
The last convolutional layer reduces the number of channels to 1, generating a single-channel output, which represents the discriminator's decision regarding the input image.
Forward Method (forward):

def forward(self, x):: Defines the forward pass of the model, specifying how input data x is processed through the layers defined in self.model.
return self.model(x): Returns the output of the model after processing the input x through all the layers defined in the self.model.
In summary, this Discriminator class defines a convolutional neural network architecture for image discrimination, which takes an image tensor as input and produces a single-channel output representing the discriminator's confidence score for the input image being real or fake.
"""
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.scale_factor = 16

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)

#CycleGAN Generator -----------------------------------------------
"""
This code defines a residual block, which is a fundamental building block in many deep learning architectures, particularly in neural networks designed for tasks like image generation or style transfer. Here's a breakdown of the code:

Class Definition:

class ResidualBlock(nn.Module):: Defines a class named ResidualBlock that inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
Initialization Method (__init__):

def __init__(self, in_channels):: Constructor method for the ResidualBlock class. It takes one parameter in_channels, which specifies the number of input channels.
super(ResidualBlock, self).__init__(): Calls the constructor of the parent class nn.Module to initialize the ResidualBlock class.
self.block = nn.Sequential(...): Defines the architecture of the residual block using a sequential container nn.Sequential, which sequentially stacks layers.
Residual Block Architecture (nn.Sequential):

The residual block consists of two sets of operations, each containing a reflection padding, a convolutional layer, instance normalization, and a ReLU activation function.
nn.ReflectionPad2d: Reflection padding layer, used to pad the input tensor symmetrically.
nn.Conv2d: 2D convolutional layer.
nn.InstanceNorm2d: Instance normalization layer, which normalizes each channel of the input separately for each sample in a batch.
nn.ReLU: Rectified Linear Unit activation function.
The first convolutional layer in each set has a kernel size of 3x3, while the reflection padding ensures that the spatial dimensions of the feature maps remain the same.
The ReLU activation function introduces non-linearity into the residual block.
The final operation in the residual block is an element-wise addition (x + self.block(x)), which adds the original input tensor x to the output of the residual block. This is the essence of residual learning, where the model learns to predict residuals (differences) instead of directly predicting the desired output.
Forward Method (forward):

def forward(self, x):: Defines the forward pass of the residual block, specifying how input data x is processed through the layers defined in self.block.
return x + self.block(x): Returns the sum of the input x and the output of the residual block (self.block(x)). This operation implements the residual connection, allowing the network to learn residual mappings, which are then added back to the input to produce the final output.
In summary, this ResidualBlock class defines a building block with skip connections, enabling the network to effectively learn residual mappings and facilitate the training of deeper neural networks with improved gradient flow and feature reuse.
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        self.initial = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels, 64, 2 * in_channels + 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.downsample_blocks = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residual_blocks)])

        self.upsample_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(64, in_channels, 2 * in_channels + 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsample_blocks(x)
        x = self.residual_blocks(x)
        x = self.upsample_blocks(x)
        return self.output(x)


#Training Loop 
"""
This code seems to be setting up the training loop for a model architecture aimed at image-to-image translation or similar tasks. Specifically, it creates two pairs of generators and discriminators. Here's a breakdown:

Generator and Discriminator Initialization:

G_XY = GeneratorResNet(3, num_residual_blocks=9): This line initializes a generator model called G_XY using the GeneratorResNet class. It specifies that the input to the generator will have 3 channels (presumably representing RGB images), and it uses 9 residual blocks. This generator is typically used to translate images from domain X to domain Y.

D_Y = Discriminator(3): Here, a discriminator model called D_Y is initialized using the Discriminator class. It specifies that the input to the discriminator will have 3 channels. This discriminator is used to discriminate between real and generated images in domain Y.
"""

G_XY = GeneratorResNet(3, num_residual_blocks=9)
D_Y = Discriminator(3)

G_YX = GeneratorResNet(3, num_residual_blocks=9)
D_X = Discriminator(3)


# Check for CUDA availability and define the Tensor type accordingly
"""
cuda_available = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda_available else torch.Tensor
print(f'CUDA Available: {cuda_available}')

# Transfer models to CUDA if available
models = [G_XY, D_Y, G_YX, D_X]
for model in models:
    if cuda_available:
        model.cuda()

gan_loss = nn.MSELoss().cuda()
cycle_loss = nn.L1Loss().cuda()
identity_loss = nn.L1Loss().cuda()
"""
Tensor =  torch.Tensor

optimizer_G = torch.optim.Adam(itertools.chain(G_XY.parameters(), G_YX.parameters()), lr=lr, betas=(beta1, beta2))
optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=lr, betas=(beta1, beta2))


def learning_rate_decay(epoch, decay_start_epoch, total_epochs):
    if epoch < decay_start_epoch:
        return 1
    else:
        return 1 - (epoch - decay_start_epoch) / (total_epochs - decay_start_epoch)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: learning_rate_decay(epoch, decay_epoch, n_epoches))
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=lambda epoch: learning_rate_decay(epoch, decay_epoch, n_epoches))
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=lambda epoch: learning_rate_decay(epoch, decay_epoch, n_epoches))

"""

This code defines a function sample_images that visualizes real and generated images from two domains (X and Y) using a trained image-to-image translation model. Here's an explanation of the code:

Function Definition:

def sample_images(real_X, real_Y):: Defines a function named sample_images that takes two arguments: real_X (images from domain X) and real_Y (images from domain Y).
Evaluation Mode:

G_XY.eval() and G_YX.eval(): Sets the generator models G_XY and G_YX to evaluation mode using the .eval() method. This disables dropout and batch normalization layers since they behave differently during inference than during training.
Image Generation:

real_X = real_X.type(Tensor): Converts the input images real_X to the appropriate data type (Tensor). This is necessary to feed them into the generator model.
fake_Y = G_XY(real_X).detach(): Passes the real images from domain X (real_X) through the generator G_XY to generate fake images in domain Y (fake_Y). The .detach() method is used to detach the fake images from the computational graph, preventing gradients from flowing back to the generator during visualization.
Similar operations are performed for real_Y and fake_X using the generator G_YX.
Visualization:

make_grid: PyTorch utility function used to create a grid of images.
plt.subplots(2, 2, figsize=(8, 8)): Creates a figure and a set of subplots. In this case, it creates a 2x2 grid of subplots for displaying the real X, fake Y, real Y, and fake X images.
Image grids are created for real X, fake Y, real Y, and fake X images using make_grid.
Images are plotted using imshow, and titles are set for each subplot.
Finally, plt.tight_layout() adjusts subplot parameters to give specified padding and spacing and plt.show() displays the plot.
Usage:

The function is called twice with different sets of real images (real_X and real_Y) obtained from the test_loader. This suggests that the function is used to visualize the image translation results for two different batches of test data.
Overall, this function provides a visual inspection of the image translation results between two domains using the trained generator models. It helps in understanding the quality of the generated images and assessing the performance of the image-to-image translation model.
"""
def sample_images(real_X, real_Y):    
    G_XY.eval()
    G_YX.eval()

    real_X = real_X.type(Tensor)
    fake_Y = G_XY(real_X).detach()

    real_Y = real_Y.type(Tensor)
    fake_X = G_YX(real_Y).detach()

    ncols = real_X.size(0)
    real_X_grid = make_grid(real_X, nrow=ncols, normalize=True)
    fake_Y_grid = make_grid(fake_Y, nrow=ncols, normalize=True)
    real_Y_grid = make_grid(real_Y, nrow=ncols, normalize=True)
    fake_X_grid = make_grid(fake_X, nrow=ncols, normalize=True)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))  

    axs[0, 0].imshow(real_X_grid.permute(1, 2, 0).cpu())
    axs[0, 0].set_title("Real Images from Domain X")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(fake_Y_grid.permute(1, 2, 0).cpu())
    axs[0, 1].set_title("Generated Images to Domain Y")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(real_Y_grid.permute(1, 2, 0).cpu())
    axs[1, 0].set_title("Real Images from Domain Y")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(fake_X_grid.permute(1, 2, 0).cpu())
    axs[1, 1].set_title("Generated Images to Domain X")
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

real_X, real_Y = next(iter(test_loader))
sample_images(real_X, real_Y)


real_X, real_Y = next(iter(test_loader))
sample_images(real_X, real_Y)

"""
This code snippet is a training loop for a CycleGAN model. Let's break it down:

Outer Loop (Epochs):

for epoch in range(n_epoches):: Iterates over the specified number of epochs (n_epoches). Each epoch represents a complete pass through the entire training dataset.
Inner Loop (Batches):

for i, (real_X, real_Y) in enumerate(train_loader):: Iterates over batches of training data obtained from the train_loader. real_X represents images from domain X, and real_Y represents images from domain Y.
Data Preparation:

real_X, real_Y = real_X.type(Tensor), real_Y.type(Tensor): Converts the real images to the appropriate data type (Tensor), likely to match the device (e.g., GPU) used for training.
out_shape = [...]: Defines the output shape for the discriminator predictions based on the size of the input images.
Label Preparation:

valid = torch.ones(out_shape).type(Tensor): Creates a tensor of ones (1) as the "real" label.
fake = torch.zeros(out_shape).type(Tensor): Creates a tensor of zeros (0) as the "fake" label.
Generator Training:

G_XY.train() and G_YX.train(): Sets the generator models G_XY and G_YX to training mode.
Forward pass through the generators to generate fake images (fake_Y and fake_X).
Computes various losses:
Identity loss (loss_id_X and loss_id_Y) measures how well the generators can maintain identity when translating images.
GAN loss (loss_GAN_XY and loss_GAN_YX) measures the ability of the generators to fool the discriminators.
Cycle consistency loss (loss_cycle_X and loss_cycle_Y) ensures that the translations are consistent in both directions.
Total generator loss (loss_G) combines these losses with specified weights.
Backpropagates the total generator loss and updates the generator parameters.
Discriminator Training:

Similar to the generator training, computes losses for both discriminators (loss_D_X and loss_D_Y), which include losses for real and fake images.
Backpropagates the discriminator losses and updates the discriminator parameters.
Learning Rate Scheduling:

lr_scheduler_G.step(), lr_scheduler_D_X.step(), and lr_scheduler_D_Y.step(): Adjusts the learning rates for the optimizers, typically scheduled to decay over time.
Displaying Results:

Every display_epoch (specified elsewhere), it visualizes test images and prints the losses for both generators and discriminators.
This loop trains the CycleGAN model by alternating between training the generators and the discriminators, updating their parameters iteratively to minimize the defined loss functions. After each epoch, it evaluates the model's performance on test data and provides feedback on the training progress through printed loss values.
"""

for epoch in range(n_epoches):
    for i, (real_X, real_Y) in enumerate(train_loader):
        real_X, real_Y = real_X.type(Tensor), real_Y.type(Tensor)
        out_shape = [real_X.size(0), 1, real_X.size(2) // D_X.scale_factor, real_X.size(3) // D_X.scale_factor]
        
        valid = torch.ones(out_shape).type(Tensor)
        fake = torch.zeros(out_shape).type(Tensor)
        
        # training generators
        G_XY.train()
        G_YX.train()
        
        optimizer_G.zero_grad()
        
        fake_Y = G_XY(real_X)
        fake_X = G_YX(real_Y)
        
        # identity loss
        loss_id_X = identity_loss(fake_Y, real_X)
        loss_id_Y = identity_loss(fake_X, real_Y)
        loss_identity = (loss_id_X + loss_id_Y) / 2
        
        # gan loss
        loss_GAN_XY = gan_loss(D_Y(fake_Y), valid) 
        loss_GAN_YX = gan_loss(D_X(fake_X), valid)
        loss_GAN = (loss_GAN_XY + loss_GAN_YX) / 2
        
        # cycle loss
        recov_X = G_YX(fake_Y)
        recov_Y = G_XY(fake_X)
        
        loss_cycle_X = cycle_loss(recov_X, real_X)
        loss_cycle_Y = cycle_loss(recov_Y, real_Y)
        loss_cycle = (loss_cycle_X + loss_cycle_Y) / 2
        
        # total gan loss
        loss_G = 5.0 * loss_identity + loss_GAN + 10.0 * loss_cycle
        
        loss_G.backward()
        optimizer_G.step()
        
        #training discriminator X
        optimizer_D_X.zero_grad()
        
        loss_real = gan_loss(D_X(real_X), valid)
        loss_fake = gan_loss(D_X(fake_X.detach()), fake)
        loss_D_X = (loss_real + loss_fake) / 2
        
        loss_D_X.backward()
        optimizer_D_X.step()
        
        #training discriminator Y
        optimizer_D_Y.zero_grad()
        
        loss_real = gan_loss(D_Y(real_Y), valid)
        loss_fake = gan_loss(D_Y(fake_Y.detach()), fake)
        loss_D_Y = (loss_real + loss_fake) / 2
        
        loss_D_Y.backward()
        optimizer_D_Y.step()
    
    lr_scheduler_G.step()
    lr_scheduler_D_X.step()
    lr_scheduler_D_Y.step()
    
    # display results every 20 epoch
    if (epoch + 1) % display_epoch == 0:
        test_real_X, test_real_Y = next(iter(test_loader))
        sample_images(test_real_X, test_real_Y)

        loss_D = (loss_D_X + loss_D_Y) / 2
        print(f'[Epoch {epoch + 1} / {n_epoches}]')
        print(f'[Generator loss: {loss_G.item()} | identity: {loss_identity.item()} GAN: {loss_GAN.item()} cycle: {loss_cycle.item()}]')
        print(f'[Discriminator loss: {loss_D.item()} | D_X: {loss_D_X.item()} D_Y: {loss_D_Y.item()}]')    

#Inference and Submission ----------------------------------------
files = [TEST_IMAGES_PATH + "/" + name for name in os.listdir(TEST_IMAGES_PATH)]
len(files)

save_dir = '../images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


transforms_dataset = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

tensor_to_image = transforms.ToPILImage()
G_YX.eval()

# Process images in batches
for batch_start in range(0, len(files), BATCH_SIZE):
    batch_files = files[batch_start:batch_start + BATCH_SIZE]
    batch_imgs = [transforms_dataset(Image.open(file)) for file in batch_files]
    
    batch_tensor = torch.stack(batch_imgs).type(Tensor)
    generated_imgs = G_YX(batch_tensor).detach().cpu()
    
    for idx, tensor_img in enumerate(generated_imgs):
        # Convert tensor to numpy array and normalize to range [0, 255]
        np_img = tensor_img.squeeze().permute(1, 2, 0).numpy()
        np_img = ((np_img - np_img.min()) * 255 / (np_img.max() - np_img.min())).astype(np.uint8)
        
        # Save the generated image
        pil_img = tensor_to_image(np_img)
        _, filename = os.path.split(files[batch_start + idx])
        pil_img.save(os.path.join(save_dir, filename))

shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")