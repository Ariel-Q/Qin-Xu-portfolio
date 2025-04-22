import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os

# Define the model architecture (same as in the training script)
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 4, 4)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        x_hat = torch.sigmoid(self.conv3(z))
        return x_hat


class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vector_quantizer):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vector_quantizer = vector_quantizer

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vector_quantizer(z)
        x_hat = self.decoder(quantized)
        return x_hat, loss, perplexity


# Define hyperparameters (same as in the training script)
latent_dim = 20
num_embeddings = 512
embedding_dim = 20
commitment_cost = 0.6

# Create the model
encoder = Encoder(1, 256, latent_dim)
decoder = Decoder(latent_dim, 256, 1)
vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
vqvae = VQVAE(encoder, decoder, vector_quantizer)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('./result/checkpoint/model_best.pth', map_location=device)
vqvae.load_state_dict(checkpoint['state_dict'])
vqvae.to(device)
vqvae.eval()

print("Model loaded successfully!")

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Function to extract latent vectors
def extract_latent_vectors(model, dataloader, device):
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            z = model.encoder(x)
            _, quantized, _, _ = model.vector_quantizer(z)
            latent_vectors.append(quantized.cpu().numpy())
            labels.extend(y.numpy())
    return np.concatenate(latent_vectors), np.array(labels)

# Extract latent vectors for the test dataset
print("Extracting latent vectors...")
latent_vectors, labels = extract_latent_vectors(vqvae, test_loader, device)

# Create a DataFrame with the latent vectors and data indices
df = pd.DataFrame(latent_vectors, columns=[f'dim_{i}' for i in range(latent_dim)])
df.insert(0, 'label', labels)
df.set_index('label', inplace=True)

# Save the latent vectors to a CSV file
output_dir = './result/latent_vectors'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'test_latent_vectors_pyversion.csv')
df.to_csv(output_file, index=True)

print(f"Latent vectors saved to {output_file}")
print(f"Shape of latent vectors: {latent_vectors.shape}")