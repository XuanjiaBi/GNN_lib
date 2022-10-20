#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
# from models import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee

import ipdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, DeepGraphInfomax, JumpingKnowledge
from aif360.sklearn.metrics import statistical_parity_difference as SPD
from aif360.sklearn.metrics import equal_opportunity_difference as EOD

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score
from torch.nn.utils import spectral_norm


class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Classifier, self).__init__()

        # Classifier projector
        self.fc1 = spectral_norm(nn.Linear(ft_in, nb_classes))

    def forward(self, seq):
        ret = self.fc1(seq)
        return ret


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = spectral_norm(GCNConv(nfeat, nhid))

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            spectral_norm(nn.Linear(nfeat, nhid)),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            spectral_norm(nn.Linear(nhid, nhid)),
        )
        self.conv1 = GINConv(self.mlp1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


# class JK(nn.Module):
#     def __init__(self, nfeat, nhid, dropout=0.5):
#         super(JK, self).__init__()
#         self.conv1 = spectral_norm(GCNConv(nfeat, nhid))
#         self.convx= spectral_norm(GCNConv(nhid, nhid))
#         self.jk = JumpingKnowledge(mode='max')
#         self.transition = nn.Sequential(
#             nn.ReLU(),
#         )
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, x, edge_index):
#         xs = []
#         x = self.conv1(x, edge_index)
#         x = self.transition(x)
#         xs.append(x)
#         for _ in range(1):
#             x = self.convx(x, edge_index)
#             x = self.transition(x)
#             xs.append(x)
#         x = self.jk(xs)
#         return x


# class SAGE(nn.Module):
#     def __init__(self, nfeat, nhid, dropout=0.5):
#         super(SAGE, self).__init__()
#
#         # Implemented spectral_norm in the sage main file
#         # ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
#         self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
#         self.conv1.aggr = 'mean'
#         self.transition = nn.Sequential(
#             nn.ReLU(),
#             nn.BatchNorm1d(nhid),
#             nn.Dropout(p=dropout)
#         )
#         self.conv2 = SAGEConv(nhid, nhid, normalize=True)
#         self.conv2.aggr = 'mean'
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.transition(x)
#         x = self.conv2(x, edge_index)
#         return x


class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = spectral_norm(GCNConv(nfeat, self.hidden_ch))
        self.activation = nn.PReLU()

    def corruption(self, x, edge_index):
        # corrupted features are obtained by row-wise shuffling of the original features
        # corrupted graph consists of the same nodes but located in different places
        return x[torch.randperm(x.size(0))], edge_index

    def summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class GraphInfoMax(nn.Module):
    def __init__(self, enc_dgi):
        super(GraphInfoMax, self).__init__()
        self.dgi_model = DeepGraphInfomax(enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption)

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        return pos_z


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                base_model='gcn', k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == 'gcn':
            self.conv = GCN(in_channels, out_channels)
        # elif self.base_model == 'gin':
        #     self.conv = GIN(in_channels, out_channels)
        # elif self.base_model == 'sage':
        #     self.conv = SAGE(in_channels, out_channels)
        # elif self.base_model == 'infomax':
        #     enc_dgi = Encoder_DGI(nfeat=in_channels, nhid=out_channels)
        #     self.conv = GraphInfoMax(enc_dgi=enc_dgi)
        # elif self.base_model == 'jk':
        #     self.conv = JK(in_channels, out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv(x, edge_index)
        return x


class SSF(torch.nn.Module):

    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                sim_coeff: float = 0.5, nclass: int=1):
        super(SSF, self).__init__()
        self.encoder: Encoder = encoder
        self.sim_coeff: float = sim_coeff

        # Projection
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_proj_hidden)),
            nn.BatchNorm1d(num_proj_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(num_proj_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden)
        )

        # Prediction
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc4 = spectral_norm(nn.Linear(num_hidden, num_hidden))

        # Classifier
        self.c1 = Classifier(ft_in=num_hidden, nb_classes=nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor,
                    edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        return z

    def prediction(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        return z

    def classifier(self, z):
        return self.c1(z)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    def D_entropy(self, x1, x2):
        x2 = x2.detach()
        return (-torch.max(F.softmax(x2), dim=1)[0]*torch.log(torch.max(F.softmax(x1), dim=1)[0])).mean()

    def D(self, x1, x2): # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, e_1, e_2, idx):

        # projector
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        # predictor
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)

        # classifier
        c1 = self.classifier(z1)

        l1 = self.D(h1[idx], p2[idx])/2
        l2 = self.D(h2[idx], p1[idx])/2
        l3 = F.cross_entropy(c1[idx], z3[idx].squeeze().long().detach())

        return self.sim_coeff*(l1+l2), l3

    def fair_metric(self, pred, labels, sens):
        idx_s0 = sens==0
        idx_s1 = sens==1

        idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)

        parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))

        return parity.item(), equality.item()

    def predict(self, emb):

        # projector
        p1 = self.projection(emb)

        # predictor
        h1 = self.prediction(p1)

        # classifier
        c1 = self.classifier(emb)

        return c1

    def linear_eval(self, emb, labels, idx_train, idx_test):
        x = emb.detach()
        classifier = nn.Linear(in_features=x.shape[1], out_features=2, bias=True)
        classifier = classifier.to('cuda')
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
        for i in range(1000):
            optimizer.zero_grad()
            preds = classifier(x[idx_train])
            loss = F.cross_entropy(preds, labels[idx_train])
            loss.backward()
            optimizer.step()
            if i%100==0:
                print(loss.item())
        classifier.eval()
        preds = classifier(x[idx_test]).argmax(dim=1)
        correct = (preds == labels[idx_test]).sum().item()
        return preds, correct/preds.shape[0]

    def fit(self,best_loss, features, edge_index, labels):
        for epoch in range(args.epochs + 1):
            # t = time.time()
            if args.model == 'ssf':
                sim_loss = 0
                cl_loss = 0
                rep = 1
                for _ in range(rep):
                    model.train()
                    optimizer_1.zero_grad()
                    optimizer_2.zero_grad()
                    edge_index_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
                    edge_index_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]
                    x_1 = drop_feature(features, args.drop_feature_rate_2, sens_idx, sens_flag=False)
                    x_2 = drop_feature(features, args.drop_feature_rate_2, sens_idx)
                    z1 = model(x_1, edge_index_1)
                    z2 = model(x_2, edge_index_2)

                    # projector
                    p1 = model.projection(z1)
                    p2 = model.projection(z2)

                    # predictor
                    h1 = model.prediction(p1)
                    h2 = model.prediction(p2)

                    l1 = model.D(h1[idx_train], p2[idx_train]) / 2
                    l2 = model.D(h2[idx_train], p1[idx_train]) / 2
                    sim_loss += args.sim_coeff * (l1 + l2)

                (sim_loss / rep).backward()
                optimizer_1.step()

                # classifier
                z1 = model(x_1, edge_index_1)
                z2 = model(x_2, edge_index_2)
                c1 = model.classifier(z1)
                c2 = model.classifier(z2)

                # Binary Cross-Entropy
                l3 = F.binary_cross_entropy_with_logits(c1[idx_train],
                                                        labels[idx_train].unsqueeze(1).float().to(device)) / 2
                l4 = F.binary_cross_entropy_with_logits(c2[idx_train],
                                                        labels[idx_train].unsqueeze(1).float().to(device)) / 2

                cl_loss = (1 - args.sim_coeff) * (l3 + l4)
                cl_loss.backward()
                optimizer_2.step()
                loss = (sim_loss / rep + cl_loss)

                # Validation
                model.eval()
                val_s_loss, val_c_loss = ssf_validation(model, val_x_1, val_edge_index_1, val_x_2, val_edge_index_2,
                                                        labels)
                emb = model(val_x_1, val_edge_index_1)
                output = model.predict(emb)
                preds = (output.squeeze() > 0).type_as(labels)
                auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])

                # if epoch % 100 == 0:
                #     print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/rep):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {auc_roc_val:.4f}")

                if (val_c_loss + val_s_loss) < best_loss:
                    # print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
                    best_loss = val_c_loss + val_s_loss
                    torch.save(model.state_dict(), f'weights_ssf_{args.encoder}.pt')
        print("Optimization Finished!")

def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1-x[:, sens_idx]

    return x

def ssf_validation(model, x_1, edge_index_1, x_2, edge_index_2, y):
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1[idx_val], p2[idx_val])/2
    l2 = model.D(h2[idx_val], p1[idx_val])/2
    sim_loss = args.sim_coeff*(l1+l2)

    # classifier
    c1 = model.classifier(z1)
    c2 = model.classifier(z2)

    # Binary Cross-Entropy
    l3 = F.binary_cross_entropy_with_logits(c1[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2
    l4 = F.binary_cross_entropy_with_logits(c2[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2

    return sim_loss, l3+l4


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--proj_hidden', type=int, default=16,
                    help='Number of hidden units in the projection layer of encoder.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--drop_edge_rate_1', type=float, default=0.1,
                    help='drop edge for first augmented graph')
parser.add_argument('--drop_edge_rate_2', type=float, default=0.1,
                    help='drop edge for second augmented graph')
parser.add_argument('--drop_feature_rate_1', type=float, default=0.1,
                    help='drop feature for first augmented graph')
parser.add_argument('--drop_feature_rate_2', type=float, default=0.1,
                    help='drop feature for second augmented graph')
parser.add_argument('--sim_coeff', type=float, default=0.5,
                    help='regularization coeff for the self-supervised task')
parser.add_argument('--dataset', type=str, default='loan',
                    choices=['nba','bail','loan', 'credit', 'german'])
parser.add_argument("--num_heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn'])
parser.add_argument('--encoder', type=str, default='gcn')
print("parser: ", parser)

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
# print(args.dataset)

# Load credit_scoring dataset
if args.dataset == 'credit':
	sens_attr = "Age"  # column number after feature process is 1
	sens_idx = 1
	predict_attr = 'NoDefaultNextMonth'
	label_number = 6000
	path_credit = "dataset/credit"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_credit,
	                                                                        label_number=label_number
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features

# Load german dataset
elif args.dataset == 'german':
	sens_attr = "Gender"  # column number after feature process is 0
	sens_idx = 0
	predict_attr = "GoodCustomer"
	label_number = 100
	path_german = "dataset/german"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_german,
	                                                                        label_number=label_number,
	                                                                        )
# Load bail dataset
elif args.dataset == 'bail':
	sens_attr = "WHITE"  # column number after feature process is 0
	sens_idx = 0
	predict_attr = "RECID"
	label_number = 100
	path_bail = "dataset/bail"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(args.dataset, sens_attr,
																			predict_attr, path=path_bail,
	                                                                        label_number=label_number,
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features
else:
	print('Invalid dataset name!!')
	exit(0)

edge_index = convert.from_scipy_sparse_matrix(adj)[0]

#%%
# Model and optimizer
num_class = labels.unique().shape[0]-1
print("Running model: ",args.model)

if args.model == 'ssf':
	encoder = Encoder(in_channels=features.shape[1], out_channels=args.hidden, base_model=args.encoder).to(device)
	model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff, nclass=num_class).to(device)
	val_edge_index_1 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_1)[0]
	val_edge_index_2 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_2)[0]
	val_x_1 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, sens_flag=False)
	val_x_2 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx)
	par_1 = list(model.encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters()) + list(model.fc4.parameters())
	par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
	optimizer_1 = optim.Adam(par_1, lr=args.lr, weight_decay=args.weight_decay)
	optimizer_2 = optim.Adam(par_2, lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)
else:
    print("invalid model")

# Train model

t_total = time.time()
best_loss = 100
best_acc = 0
features = features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)

##### Code: ########
# fit(best_loss)
if args.model == 'ssf':
    model.fit(best_loss, features, edge_index, labels)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))



model.load_state_dict(torch.load(f'weights_ssf_{args.encoder}.pt'))
model.eval()
emb = model(features.to(device), edge_index.to(device))
output = model.predict(emb)
counter_features = features.clone()
counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
counter_output = model.predict(model(counter_features.to(device), edge_index.to(device)))
noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
noisy_output = model.predict(model(noisy_features.to(device), edge_index.to(device)))

# Report
output_preds = (output.squeeze()>0).type_as(labels)
counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
noisy_output_preds = (noisy_output.squeeze()>0).type_as(labels)
auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])

parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())

# print report
print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
print(f'Parity: {parity} | Equality: {equality}')
print(f'F1-score: {f1_s}')
print(f'CounterFactual Fairness: {counterfactual_fairness}')
print(f'Robustness Score: {robustness_score}')
