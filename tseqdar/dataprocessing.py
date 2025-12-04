import numpy as np
import torch
from torch_geometric.data import Data
import mdtraj as md
import pickle, glob, os
from tqdm import tqdm   
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def average_position(positions, box):
    box_inv = np.linalg.inv(box)
    frac = positions @ box_inv.T  # shape (N, 3)
    frac0 = frac[0]
    delta = frac - frac0
    delta -= np.round(delta)  
    frac_unwrapped = frac0 + delta
    center_frac = frac_unwrapped.mean(axis=0)
    center_cart = center_frac @ box.T
    return center_cart

def find_cg_pos(ca_pos,cg_degree,box,dic_chain,node_chain):
    """
    Calculate coarse-grained positions and indices.
    Accepts/returns cartesian coordinates while operates on reciprocal space .
    """
    cg_pos = []
    cg_index = []
    count = 0
    for i in dic_chain.keys():
        chain_mask = np.where(node_chain == dic_chain[i])[0]
        chain_pos = ca_pos[chain_mask]
        for j in range(len(chain_pos)//cg_degree+int(len(chain_pos)%cg_degree !=0)):
            window = chain_pos[j*cg_degree:(j+1)*cg_degree]
            cg_index.extend([count]*len(window))
            count += 1
            if len(window) == 1:
                cg_pos.append(window[0])
            else:
                cg_pos.append(average_position(window,box))
    return np.array(cg_pos), np.array(cg_index)

def periodic_boundary(diff, box):
    box_inv = torch.linalg.inv(box)
    diff_frac =  torch.einsum('ij,nmj->nmi', box_inv, diff)
    diff_frac -= torch.round(diff_frac)
    diff = torch.einsum('ij,nmj->nmi', box, diff_frac)
    return diff 

def find_neighbors(pos, box, cutoff, max_num_neighbors=20, loop=True, direction="source_to_target"):
    N = pos.size(0)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 3]
    diff = periodic_boundary(diff, box)
    dist2 = (diff ** 2).sum(dim=-1)  # [N, N]
    mask = (dist2 < cutoff ** 2)
    if not loop:
        mask &= ~torch.eye(N, dtype=torch.bool, device=pos.device)

    edge_index_list = []
    edge_weight_list = []
    diff_list = []

    for i in range(N):
        valid = mask[i]  # [N]
        d2 = dist2[i][valid]  # distances to neighbors
        j_indices = valid.nonzero(as_tuple=False).squeeze(1)  # neighbor indices

        if d2.numel() > max_num_neighbors:
            topk = torch.topk(d2, max_num_neighbors, largest=False)
            j_indices = j_indices[topk.indices]
            d2 = d2[topk.indices]

        i_indices = torch.full_like(j_indices, i)

        edge_index_list.append(torch.stack([i_indices, j_indices], dim=0))
        edge_weight_list.append(d2.sqrt())
        diff_list.append(diff[i, j_indices])

    edge_index = torch.cat(edge_index_list, dim=1)  # [2, E]
    edge_weight = torch.cat(edge_weight_list)       # [E]
    displacement = torch.cat(diff_list, dim=0)      # [E, 3]

    if direction == "source_to_target":
        pass  
    elif direction == "target_to_source":
        edge_index = edge_index[[1, 0]]
        displacement = -displacement
    else:
        raise ValueError("Invalid direction")

    return edge_index, edge_weight, displacement

# node embedding distinguishing chains 
z_str = []
ele_str = []
red_str = []
chain_str = []
with open('./fixed.pdb') as f:
    for line in f:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:17].strip()
            ele_name = line[77:78].strip()
            red_name = line[17:20].strip()
            chain_name = line[21:22].strip()
            red_str.append(red_name)
            z_str.append(atom_name)
            ele_str.append(ele_name)
            chain_str.append(chain_name)

dic_atom = {e:i for i, e in enumerate(np.unique(z_str))}
z_num = np.array([dic_atom[name] for name in z_str])
dic_red = {e:i for i, e in enumerate(np.unique(red_str))}
red_num = np.array([dic_red[name] for name in red_str])
dic_ele = {e:i for i, e in enumerate(np.unique(ele_str))}
ele_num = np.array([dic_ele[name] for name in ele_str])
dic_chain = {e:i for i,e in enumerate(np.unique(chain_str))}
chain_num = np.array([dic_chain[name] for name in chain_str])

pro_red = np.array(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                    'TYR', 'VAL'])
pro_red_num = np.array([dic_red[name] for name in pro_red if name in dic_red.keys()])

all_chain = []
for chain in dic_chain.keys():
    chain_id = dic_chain[chain]
    chain_mask = (chain_num == chain_id) # atomic
    chain_red = red_num[chain_mask] # atomic
    if not np.any(np.isin(np.unique(chain_red), pro_red_num)): # permits nonstandard residues
        chain_mask &= (ele_num != dic_ele['H'])  # heavy atom
        chain_emb = ele_num[chain_mask]
        mode = 'heavy'
    else:
        chain_mask &= (z_num == dic_atom['CA'])  # CA atom
        chain_emb = red_num[chain_mask]
        mode = 'CA'
    print(f"chain: {chain}, num_atoms: {len(chain_emb)}")
    # differenate chains
    all_chain.append({
        'chain id': chain_id,
        'selection mode': mode,
        'chain embed': chain_emb,
        'chain selection mask': chain_mask
    })

node_embed = []
node_mask = np.zeros(len(z_num), dtype=bool)
for chain in all_chain:
    if len(node_embed) > 0:
        node_embed.extend(chain['chain embed']+np.max(node_embed)+1)
    else:
        node_embed.extend(chain['chain embed'])
    node_mask += chain['chain selection mask']
node_embed = np.array(node_embed)
node_mask = np.array(node_mask)
node_chain = chain_num[node_mask]

# construct molecular graph
cutoff = 1.2
cg_degree = 3
files = list(glob.glob(os.path.join('./traj_dt1ns', "*.xtc")))

for i,file in tqdm(enumerate(files)):
    traj = md.load(file,top='./fixed.pdb')
    traj_data = []
    pos = traj.xyz
    for xyz in pos:
        sele_pos = xyz[node_mask]
        cg_pos, cg_index = find_cg_pos(sele_pos,cg_degree,box=traj.unitcell_vectors[0],dic_chain=dic_chain,node_chain=node_chain)
        edge_index, edge_weight, edge_vec = find_neighbors(torch.tensor(cg_pos),torch.tensor(traj.unitcell_vectors[0]), cutoff)
        traj_data.append(Data(z=torch.tensor(node_embed),
                              pos=torch.tensor(cg_pos),
                              cg_index = torch.tensor(cg_index),
                              edge_index = edge_index,
                              edge_weight = edge_weight,
                              edge_vec = edge_vec
                              ))
    with open('./CG3_cut12/traj_%i.pickle'%i,'wb') as f:
        pickle.dump(traj_data,f)