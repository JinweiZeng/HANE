from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import os
import multiprocessing as mp
from torch.utils.data import Dataset
import os
import random

os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import json

import dgl
import dgl.data
from torch.utils.data.sampler import SubsetRandomSampler

from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv, MaxPooling, GlobalAttentionPooling
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from dgl.data.utils import save_graphs, load_graphs
from torch.utils.tensorboard import SummaryWriter
import geopandas as gpd
from shapely.geometry import mapping
from dgl.nn import SumPooling, GlobalAttentionPooling, AvgPooling, MaxPooling, SortPooling, WeightAndSum
from dgl.nn.pytorch.conv import NNConv, EGATConv, GATConv
from sklearn.model_selection import train_test_split
import time
import dgl.nn.pytorch as dglnn
import setproctitle
import multiprocessing as mp
import warnings
from torch.utils.data import Dataset, DataLoader

# time.sleep(24*60*60)

warnings.filterwarnings('ignore')
setproctitle.setproctitle('gcn_pred_cross_county@zengjinwei')

parser = argparse.ArgumentParser(description='Carbon_Prediction')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--coef', type=float, default=1)
parser.add_argument('--countyOD', type=int, default=0)
parser.add_argument('--rncount', type=int, default=2)
parser.add_argument('--cbgcount', type=int, default=2)
parser.add_argument('--countycount', type=int, default=2)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--cbgodfiltered', type=int, default=0)

args = parser.parse_args()
writer = SummaryWriter(
    'new_dataset/gcn_pred_cbg_dual_cross_county_scaleis{}/{}_{}_{}_{}_nodelayeris{}_cbglayeris{}_countylayeris_{}'.format(str(args.scale), str(args.lr),
                                                                           str(args.batch_size), str(args.coef), str(args.countyOD), str(args.rncount), str(args.cbgcount), str(args.countycount)))
writer.add_scalar('lr', args.lr)
writer.add_scalar('batch_size', args.batch_size)

seed = 23
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)


class RoadNetworkDataset(Dataset):
    def __init__(self):
        graph_list, graph_labels = load_graphs('node_edge_files/gnn_dataset_usa_counties_three_layers_cross_county.bin')

        print('graph count: ', len(graph_list) / 3)

        county_graph_list = graph_list[2::3]
        cbg_graph_list = graph_list[1::3]
        graph_list = graph_list[::3]

        # 读取mapping dict
        cbg2node_list = np.load('new_dataset/mapping_dict/cbg2node_list.npy', allow_pickle=True).item()
        county2cbg_list = np.load('new_dataset/mapping_dict/county2cbg_list.npy', allow_pickle=True).item()
        state2county_list = np.load('new_dataset/mapping_dict/state2county_list.npy', allow_pickle=True).item()
        county2edge_list = np.load('new_dataset/mapping_dict/county2edge_list.npy', allow_pickle=True).item()

        # 归一化
        bg = dgl.batch(graph_list)
        max_evalue = torch.max(bg.edata['feats'], 0).values
        min_evalue = torch.min(bg.edata['feats'], 0).values
        max_nvalue = torch.max(bg.ndata['feats'], 0).values
        min_nvalue = torch.min(bg.ndata['feats'], 0).values

        for m in range(len(graph_list)):
            graph_list[m].edata['feats'] = (graph_list[m].edata['feats'] - min_evalue) / (max_evalue - min_evalue)
            graph_list[m].ndata['feats'] = (graph_list[m].ndata['feats'] - min_nvalue) / (max_nvalue - min_nvalue)

        bg = dgl.batch(cbg_graph_list)
        max_evalue = torch.max(bg.edata['feats'], 0).values
        min_evalue = torch.min(bg.edata['feats'], 0).values

        for m in range(len(cbg_graph_list)):
            cbg_graph_list[m].edata['feats'] = (cbg_graph_list[m].edata['feats'] - min_evalue) / (
                        max_evalue - min_evalue)

        bg = dgl.batch(county_graph_list)
        max_evalue = torch.max(bg.edata['feats'], 0).values
        min_evalue = torch.min(bg.edata['feats'], 0).values

        for m in range(len(county_graph_list)):
            county_graph_list[m].edata['feats'] = (county_graph_list[m].edata['feats'] - min_evalue) / (
                        max_evalue - min_evalue)

        state_fip_list = [str(m).zfill(2) for m in graph_labels['gfip'].numpy()]
        state2graph = dict(zip(state_fip_list, graph_list))
        state2cbggraph = dict(zip(state_fip_list, cbg_graph_list))
        state2countygraph = dict(zip(state_fip_list, county_graph_list))

        data = np.load('rf_dataset_all_emissions_od_filtered_darte.npy', allow_pickle=True).item()
        labels = np.array(data['labels'])
        all_idxs = np.array(list(range(len(labels))))
        valid_idxs = all_idxs[labels>0]
        num_examples = len(valid_idxs)
        print('Num of Samples:', num_examples)

        fips_list_final = np.load('new_dataset/fip_list_final.npy', allow_pickle=True)
        train_idxs, test_idxs = train_test_split(list(range(num_examples)), train_size=0.8, random_state=seed)
        train_idxs, val_idxs = train_test_split(train_idxs, train_size=0.75, random_state=seed)

        train_idxs = valid_idxs[train_idxs]
        val_idxs = valid_idxs[val_idxs]
        test_idxs = valid_idxs[test_idxs]

        train_fips = np.array(fips_list_final)[train_idxs].tolist()
        val_fips = np.array(fips_list_final)[val_idxs].tolist()
        test_fips = np.array(fips_list_final)[test_idxs].tolist()

        # darte emission
        emissions = pd.read_csv('new_dataset/DARTE_road_co2.csv', dtype={'county':'str'})
        emissions['county'] = emissions.apply(lambda row: str(row['county']).zfill(5), axis=1)
        emissions['kgco2_2017'] = emissions['kgco2_2017'].apply(lambda x: np.log(max(x, 1)))
        county2carbon = dict(zip(list(emissions['county']), list(emissions['kgco2_2017'])))

        self.county2carbon = county2carbon
        self.state_list = state_fip_list
        self.graph_list = graph_list
        self.cbg_graph_list = cbg_graph_list
        self.county_graph_list = county_graph_list
        self.state2graph = state2graph
        self.state2cbggraph = state2cbggraph
        self.state2countygraph = state2countygraph
        self.dim_nfeats = graph_list[0].ndata['feats'].shape[1]
        self.dim_efeats = graph_list[0].edata['feats'].shape[1]
        self.dim_cbgefeats = cbg_graph_list[0].edata['feats'].shape[1]
        self.dim_countyefeats = county_graph_list[0].edata['feats'].shape[1]
        print(self.dim_nfeats, self.dim_efeats, self.dim_cbgefeats, self.dim_countyefeats)

        # 生成pooling矩阵：稀疏
        county2cbg_matrix_list = {}
        cbg2node_matrix_list = {}
        county2edge_matrix_list = {}
        for key in county2cbg_list:
            county2cbg = county2cbg_list[key]
            indices = []
            for county in county2cbg:
                indices += [[county, value] for value in county2cbg[county]]
            values = np.ones(len(indices)).tolist()
            indices = [np.array(indices)[:, 0].tolist(), np.array(indices)[:, 1].tolist()]
            pool_matrix = torch.sparse_coo_tensor(indices, values, size=(len(county2cbg), 1+max(indices[1])))
            county2cbg_matrix_list[key] = pool_matrix
        for key in cbg2node_list:
            cbg2node = cbg2node_list[key]
            indices = []
            for cbg in cbg2node:
                indices += [[cbg, value] for value in cbg2node[cbg]]
            values = np.ones(len(indices)).tolist()
            indices = [np.array(indices)[:, 0].tolist(), np.array(indices)[:, 1].tolist()]
            pool_matrix = torch.sparse_coo_tensor(indices, values, size=(len(cbg2node), 1+max(indices[1])))
            cbg2node_matrix_list[key] = pool_matrix
        for key in county2edge_list:
            county2edge = county2edge_list[key]
            indices = []
            for county in county2edge:
                indices += [[county, value] for value in county2edge[county]]
            values = np.ones(len(indices)).tolist()
            indices = [np.array(indices)[:, 0].tolist(), np.array(indices)[:, 1].tolist()]
            pool_matrix = torch.sparse_coo_tensor(indices, values, size=(len(county2edge), 1+max(indices[1])))
            county2edge_matrix_list[key] = pool_matrix
        self.county2cbg_matrix_list = county2cbg_matrix_list
        self.cbg2node_matrix_list = cbg2node_matrix_list
        self.county2edge_matrix_list = county2edge_matrix_list

        state2idxs = {}
        for key in county2cbg_list:
            idxs = {'train': [], 'val': [], 'test': []}
            state2county = state2county_list[key]
            for fip in state2county:
                if fip in train_fips:
                    idxs['train'].append(state2county[fip])
                if fip in val_fips:
                    idxs['val'].append(state2county[fip])
                if fip in test_fips:
                    idxs['test'].append(state2county[fip])
            state2idxs[key] = idxs

        train_idxs = {key:state2idxs[key]['train'] for key in state2idxs}
        self.train_idxs = train_idxs

        state2labels = {}
        for key in county2cbg_list:
            state2county = state2county_list[key]
            labels = np.zeros(len(state2county))
            for fip in state2county:
                labels[state2county[fip]] = self.county2carbon[fip]
            state2labels[key] = labels

        self.idxs = state2idxs
        self.labels = state2labels

    def __len__(self):
        return len(self.state_list)

    def __getitem__(self, args):  # 输出该id对应的state_graph, cbg_graph, cbg2node, county2cbg, label
        idx = args[0]  # state_id
        method = args[1]
        graph = self.graph_list[idx]
        cbg_graph = self.cbg_graph_list[idx]
        county_graph = self.county_graph_list[idx]
        state = self.state_list[idx]
        idxs = torch.tensor(self.idxs[state][method]).long()
        label = torch.tensor(self.labels[state])
        return graph, cbg_graph, county_graph, label, state, idxs
    
    def get_graph(self, state_id):
        return self.state2graph[state_id], self.state2cbggraph[state_id], self.state2countygraph[state_id], torch.tensor(self.labels[state_id])

dataset = RoadNetworkDataset()
county2cbg_matrix_list = dataset.county2cbg_matrix_list
cbg2node_matrix_list = dataset.cbg2node_matrix_list
county2edge_matrix_list = dataset.county2edge_matrix_list

num_examples = len(dataset)
print('Num of Samples:', num_examples)

state_idxs = list(range(num_examples))
dataloader = DataLoader(state_idxs, batch_size=1, shuffle=True)

class MPNNNet(nn.Module):
    def __init__(self, in_nfeats, hidden_nfeats, in_efeats, hidden_efeats, od_efeats, countyodfeats, num_heads):
        super(MPNNNet, self).__init__()
        self.hidden_feats = hidden_nfeats
        self.conv1 = EGATConv(in_node_feats=in_nfeats, in_edge_feats=in_efeats, out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)
        self.conv2 = EGATConv(in_node_feats=hidden_nfeats*3, in_edge_feats=hidden_efeats*3, out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)
        self.conv3 = EGATConv(in_node_feats=hidden_nfeats*3, in_edge_feats=od_efeats, out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)
        self.conv4 = EGATConv(in_node_feats=hidden_nfeats*3, in_edge_feats=hidden_efeats*3, out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)

    def forward(self, g, cbg_g, county_g, nfeat, efeat, odfeat, countyodfeats, state, idxs):
        nh, eh = self.conv1(g, nfeat, efeat)
        nh = F.relu(torch.flatten(nh,1))
        eh = F.relu(torch.flatten(eh,1))

        nh, eh = self.conv2(g, nh, eh)
        g.ndata['h'] = torch.flatten(nh,1)
        g.edata['h'] = torch.flatten(eh,1)

        # 写pooling矩阵
        def get_cbg2node(nhg, state):
            pool_matrix = cbg2node_matrix_list[state].to(args.device)
            nhg = torch.sparse.mm(pool_matrix, nhg)
            count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()
            nhcbg = nhg/count
            pool_matrix_county = county2cbg_matrix_list[state].to(args.device)
            nhcountyg = torch.sparse.mm(pool_matrix_county, nhcbg)
            count = torch.sparse.sum(pool_matrix_county, 1).unsqueeze(1).to_dense()
            nhcountyg = nhcountyg/count
            pool_matrix = pool_matrix.detach().cpu()
            pool_matrix_county = pool_matrix_county.detach().cpu()
            del pool_matrix, pool_matrix_county
            return nhcbg.to(args.device).float(), nhcountyg.to(args.device).float()

        cbg_g.ndata['h'], nhcountyg = get_cbg2node(g.ndata['h'], state)
        
        nh, eh = self.conv3(cbg_g, cbg_g.ndata['h'], odfeat)
        nh = F.relu(torch.flatten(nh,1))
        eh = F.relu(torch.flatten(eh,1))

        nh, eh = self.conv4(cbg_g, nh, eh)
        cbg_g.ndata['h'] = torch.flatten(nh,1)
        cbg_g.edata['h'] = torch.flatten(eh,1)

        def get_county2cbg(cbgnh, state):
            pool_matrix = county2cbg_matrix_list[state].to(args.device)
            # 稀疏矩阵算法
            nhcbg = torch.sparse.mm(pool_matrix, cbgnh)
            count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()
            nhcbg = nhcbg/count
            pool_matrix = pool_matrix.detach().cpu()
            return nhcbg

        nhcbg = get_county2cbg(cbg_g.ndata['h'], state).float()
        county_g.ndata['h'] = nhcbg

        # 加入edge embedding的pooling结果
        def get_county2edge(eh, state):
            pool_matrix = county2edge_matrix_list[state].to(args.device)
            ehcounty = torch.sparse.mm(pool_matrix, eh)
            count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()
            ehcounty = ehcounty/count
            pool_matrix = pool_matrix.detach().cpu()
            return ehcounty

        ehcounty = get_county2edge(g.edata['h'], state).float()
        return county_g.ndata['h'][idxs, :], nhcountyg[idxs, :], ehcounty[idxs, :], nhcbg[idxs, :], county_g
    
class COUNTYOD(nn.Module):
    def __init__(self, hidden_nfeats, hidden_efeats, countyodfeats, num_heads):
        super(COUNTYOD, self).__init__()
        self.conv1 = EGATConv(in_node_feats=hidden_nfeats * 3, in_edge_feats=countyodfeats,
                            out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)
        self.layers = nn.ModuleList()
        for i in range(args.countycount-1):
            self.layers.append(EGATConv(in_node_feats=hidden_nfeats * 3, in_edge_feats=hidden_efeats * 3,
                            out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads))

    def forward(self, county_g, countyodfeats, idxs):
        nh, eh = self.conv1(county_g, county_g.ndata['h'], countyodfeats)
        nh = F.relu(torch.flatten(nh, 1))
        eh = F.relu(torch.flatten(eh, 1))
        for i in range(args.countycount-1):
            layer = self.layers[i]
            nh, eh = layer(county_g, nh, eh)
            if i != args.countycount-2:
                nh = F.relu(torch.flatten(nh, 1))
                eh = F.relu(torch.flatten(eh, 1))

        county_g.ndata['h'] = torch.flatten(nh,1)
        county_g.edata['h'] = torch.flatten(eh,1)
        return county_g.ndata['h'][idxs, :]


class Regressor(nn.Module):
    def __init__(self, in_feats, h_feats, in_efeats, h_efeats, in_odfeats, in_countyodfeats, num_heads=3):
        super(Regressor, self).__init__()
        self.gcn = MPNNNet(in_feats, h_feats, in_efeats, h_efeats, in_odfeats, in_countyodfeats, num_heads)
        self.regressor = nn.Sequential(nn.Linear(h_feats * num_heads * 3, h_feats * 2 * 2),
                nn.ReLU(),
                nn.Linear(h_feats * 2 * 2, h_feats * 2),
                nn.ReLU(),
                nn.Linear(h_feats*2, h_feats*2),
                nn.ReLU(),
                # nn.Linear(h_feats*2, h_feats*2),
                # nn.ReLU(),
                nn.Linear(h_feats * 2, h_feats),
                nn.ReLU(),
                # nn.Linear(h_feats, h_feats),
                # nn.ReLU(),
                nn.Linear(h_feats, 1))

    def forward(self, g, cbg_g, county_g, nfeat, efeat, odfeat, countyodfeat, state, idxs):
        county_embedding, node_embedding, edge_embedding, cbg_node_embedding, county_g = self.gcn(g, cbg_g, county_g, nfeat, efeat, odfeat, countyodfeat, state, idxs)
        pred = self.regressor(torch.cat([cbg_node_embedding, node_embedding, edge_embedding], dim=1))
        return pred


model = Regressor(dataset.dim_nfeats, 8, dataset.dim_efeats, 8, dataset.dim_cbgefeats, dataset.dim_countyefeats).to(args.device)
pretrained_model = torch.load('pretrained_models/noCountyOD_pretrained1.pt')
selected_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if 'gcn' in k}
model.load_state_dict(selected_dict, strict=False)

print('Lr:', args.lr)
print('Batch Size:', args.batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss = torch.nn.MSELoss(reduce='sum')
scale_loss = torch.nn.MSELoss(reduce='sum')

min_loss = -10
best_epoch = 0
n_step, t_step = 0, 0
patience = 0

def train_batch_gen(idxs):
    batch = []
    for state in idxs:
        if len(idxs[state]) == 0:
            continue
        random.shuffle(idxs[state])
        count = 0
        while args.batch_size + count < len(idxs[state]) - 1:
            batch.append([state, idxs[state][count:count+args.batch_size]])
            count += args.batch_size
        batch.append([state, idxs[state][count:]])
    random.shuffle(batch)
    return batch


for epoch in range(args.epochs):
    model.train()
    losses = 0
    scale_losses = 0
    train_preds, train_labels = [], []
    train_batch = train_batch_gen(dataset.train_idxs)
    for state, idxs in train_batch:
        graph, cbg_graph, county_graph, labels = dataset.get_graph(state)
        graph = graph.to(args.device)
        cbg_graph = cbg_graph.to(args.device)
        county_graph = county_graph.to(args.device)
        labels = labels.to(args.device)
        pred = model(graph, cbg_graph, county_graph, graph.ndata['feats'], graph.edata['feats'], cbg_graph.edata['feats'], county_graph.edata['feats'], state, idxs)
        loss_epoch = loss(pred.squeeze().float(), labels[idxs].float())
        loss_total = loss_epoch
        losses += loss_epoch.item()
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        train_preds += list(pred.cpu().detach().numpy())
        train_labels += list(labels[idxs].cpu().detach().numpy())
    mae = mean_absolute_error(train_preds, train_labels)
    rmse = np.sqrt(mean_squared_error(train_preds, train_labels))
    r_2 = r2_score(train_labels, train_preds)
    print('Training Epoch {} Pred Loss:{:.3f} MAE:{:.3f}, RMSE:{:.3f}, R2: {:.2f}'.format(epoch,
                                                                                          losses / len(train_preds),
                                                                                          mae, rmse, r_2))
    writer.add_scalar('train_pred_loss', losses / len(train_preds), n_step)
    writer.add_scalar('train_mae', mae, n_step)
    writer.add_scalar('train_rmse', rmse, n_step)
    writer.add_scalar('train_r2', r_2, n_step)
    n_step += 1

    # validation
    model.eval()
    val_preds, val_labels = [], []
    if epoch % 5 == 0:
        # val集验证
        val_losses = 0
        for idx in dataloader:
            graph, cbg_graph, county_graph, labels, state, idxs = dataset[[idx, 'val']]
            if len(idxs) == 0:
                continue
            graph = graph.to(args.device)
            cbg_graph = cbg_graph.to(args.device)
            county_graph = county_graph.to(args.device)
            labels = labels.to(args.device)
            pred = model(graph, cbg_graph, county_graph, graph.ndata['feats'], graph.edata['feats'],
                         cbg_graph.edata['feats'], county_graph.edata['feats'], state, idxs)
            loss_epoch = loss(pred.squeeze().float(), labels[idxs].float())
            loss_total = loss_epoch
            val_losses += loss_epoch.item()
            val_preds += list(pred.cpu().detach().numpy())
            val_labels += list(labels[idxs].cpu().detach().numpy())
        mae = mean_absolute_error(val_preds, val_labels)
        rmse = np.sqrt(mean_squared_error(val_preds, val_labels))
        r_2 = r2_score(val_labels, val_preds)
        print('Validation Epoch {} Pred Loss:{:.3f}, MAE:{:.3f}, RMSE:{:.3f}, R2: {:.2f}'.format(epoch,
                                                                                                 val_losses / len(
                                                                                                     val_preds), mae,
                                                                                                 rmse, r_2))
        writer.add_scalar('val_pred_loss', val_losses / len(val_preds), t_step)
        writer.add_scalar('val_mae', mae, t_step)
        writer.add_scalar('val_rmse', rmse, t_step)
        writer.add_scalar('val_R2', r_2, t_step)
        if epoch >= 50:
            patience += 1
            if r_2 > min_loss:
                patience = 0
                best_epoch = epoch
                min_loss = r_2
                fname = 'models/gnn_model_ml_dual_cross_county_{}_{}_{}_{}_nodelayeris{}_cbglayeris{}_countylayeris_{}.pt'.format(str(args.lr), str(args.batch_size), str(args.coef),
                                                                    args.scale, str(args.rncount), str(args.cbgcount), str(args.countycount))
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, fname, _use_new_zipfile_serialization=False)
            if patience > args.patience:
                print('Early Stopping at Epoch:{}'.format(epoch))
                break

        # test集也算一下
        test_preds, test_labels = [], []
        test_losses = 0
        for idx in dataloader:
            graph, cbg_graph, county_graph, labels, state, idxs = dataset[[idx, 'test']]
            if len(idxs) == 0:
                continue
            graph = graph.to(args.device)
            cbg_graph = cbg_graph.to(args.device)
            county_graph = county_graph.to(args.device)
            labels = labels.to(args.device)

            pred = model(graph, cbg_graph, county_graph, graph.ndata['feats'], graph.edata['feats'], cbg_graph.edata['feats'],
                        county_graph.edata['feats'], state, idxs)
            loss_epoch = loss(pred.squeeze().float(), labels[idxs].float())
            loss_total = loss_epoch
            test_losses += loss_epoch.item()
            test_preds += list(pred.cpu().detach().numpy())
            test_labels += list(labels[idxs].cpu().detach().numpy())
        mae = mean_absolute_error(test_preds, test_labels)
        rmse = np.sqrt(mean_squared_error(test_preds, test_labels))
        r_2 = r2_score(test_labels, test_preds)
        print('Test Epoch {} Pred Loss:{:.3f}, MAE:{:.3f}, RMSE:{:.3f}, R2: {:.2f}'.format(epoch, test_losses / len(test_preds),
                                                                                        mae, rmse, r_2))
        writer.add_scalar('test_pred_loss', test_losses / len(test_preds), t_step)
        writer.add_scalar('test_mae', mae, t_step)
        writer.add_scalar('test_rmse', rmse, t_step)
        writer.add_scalar('test_R2', r_2, t_step)
        t_step += 1

writer.add_scalar('best_epoch', best_epoch)
print('Best Epoch:', best_epoch)


# test
checkpoint = torch.load(
    'models/gnn_model_ml_dual_cross_county_{}_{}_{}_{}_nodelayeris{}_cbglayeris{}_countylayeris_{}.pt'.format(str(args.lr), str(args.batch_size), str(args.coef),
                                                                  args.scale, str(args.rncount), str(args.cbgcount), str(args.countycount)))
model.load_state_dict(checkpoint['model_state_dict'])
test_preds, test_labels = [], []
test_losses = 0
model.eval()
for idx in dataloader:
    graph, cbg_graph, county_graph, labels, state, idxs = dataset[[idx, 'test']]
    if len(idxs) == 0:
        continue
    graph = graph.to(args.device)
    cbg_graph = cbg_graph.to(args.device)
    county_graph = county_graph.to(args.device)
    labels = labels.to(args.device)
    pred = model(graph, cbg_graph, county_graph, graph.ndata['feats'], graph.edata['feats'], cbg_graph.edata['feats'],
                 county_graph.edata['feats'], state, idxs)
    loss_epoch = loss(pred.squeeze().float(), labels[idxs].float())
    loss_total = loss_epoch
    test_losses += loss_epoch.item()
    test_preds += list(pred.cpu().detach().numpy())
    test_labels += list(labels[idxs].cpu().detach().numpy())
mae = mean_absolute_error(test_preds, test_labels)
rmse = np.sqrt(mean_squared_error(test_preds, test_labels))
r_2 = r2_score(test_labels, test_preds)
print('Test Epoch {} Pred Loss:{:.3f}, MAE:{:.3f}, RMSE:{:.3f}, R2: {:.2f}'.format(epoch, test_losses / len(test_preds),
                                                                                   mae, rmse, r_2))
writer.add_scalar('test_pred_loss_final', test_losses / len(test_preds))
writer.add_scalar('test_mae_final', mae)
writer.add_scalar('test_rmse_final', rmse)
writer.add_scalar('test_R2_final', r_2)
writer.close()
