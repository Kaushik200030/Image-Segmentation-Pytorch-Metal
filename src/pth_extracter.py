import torch
# Load the .pth file
data = torch.load('/Users/kaushikpattanayak/Documents/Data Science/Machine Learning/Land-Cover-Semantic-Segmentation-PyTorch-main/models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth', map_location= torch.device('mps'))

# View the contents of the file
print(data)