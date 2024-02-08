import torch 
import time 
import scipy.sparse as sp
import scanpy as sc
import os
import ACTIVA


##CHANGE THIS PATH
path_to_pretrained = "/home/hudaa/ACTIVA/ACTIVA_68kPBMC.pth"

model_dict = torch.load(path_to_pretrained)

activa = model_dict["Saved_Model"]

print(activa)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if str(device) == "cuda":
    print('Using GPU (CUDA)')
else:
    print('Using CPU')
start = time.time()

# for reproducibility 
torch.manual_seed(1)

##CHANGE
num_cells = 68579
# look at the input size to the generator network of ACTIVA
latent_dim = 128;
z_g = torch.randn(num_cells, latent_dim).to(device)
# generate synthetic cells with ACTIVA
generated_cells = activa.decoder(z_g)

count_matrix = generated_cells.detach().cpu().numpy()

print(f"We generated {num_cells} cells in {time.time() - start} seconds (on {device})")

count_matrix_sparse = sp.csr_matrix(count_matrix)


##CHANGE THIS (same number of cells)
adata = sc.read_h5ad("/home/hudaa/Downloads/NEW8k.h5ad")

# now make the count matrix to have the same size as the generated cells (since we will replace this in the next step)

sc.pp.subsample(adata, n_obs=num_cells, random_state=0, copy=False)

adata.X = count_matrix_sparse;

dir_name = 'ACTIVA-Generated'
if not os.path.isdir(dir_name):
    os.mkdir(dir_name) 
    print(f"Created {dir_name} directory") 

path = "./" + dir_name + f"/Activa-{num_cells}Generated.h5Seurat"
adata.write(path)
print(f"Saved the new cells to {path}")
