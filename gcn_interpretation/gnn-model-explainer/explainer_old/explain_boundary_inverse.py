""" explain.py

    Implementation of the explainer. 
"""

import math
import time
import os
import sys
sys.path.append("../../ai-adversarial-detection/")

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import random

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX.utils

import torch
import torch.nn as nn
from torch.autograd import Variable

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb
import pickle

import explainer.explain as explain

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils
import utils.accuracy_utils as accuracy_utils
import utils.neighbor_utils as neighbor_utils
import utils.noise_utils as noise_utils

import dnn_invariant.extract_rules as extract

use_cuda = False# torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
hnodes_dict = pickle.load(open("/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/hnodes_dict_synthetic_train_4k8000_comb_12dlbls_nofake.p", "rb"))

n_classes = 3

incorrect_preds = 0
nbr_data = None
rule_dict_node = None

flips = 0.
inv_flips = np.zeros((n_classes))
topk_inv_flips = np.zeros((n_classes))
total = np.zeros((n_classes))
pos_diff = np.zeros((n_classes))
inv_diff = np.zeros((n_classes))
topk_inv_diff = np.zeros((n_classes))


sub_label_nodes = None
sub_label_array = None

def load_sublabel_nodes(args):
    global sub_label_nodes
    global sub_label_array

    if args.bmname == 'synthetic':
        synthetic_data = pickle.load(open(
            "/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20_12dlbls_nofake.p",
            "rb"))
    elif args.bmname == 'old_synthetic':
        synthetic_data = pickle.load(open(
            "/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20_nofake.p", "rb"))
    sub_label_nodes = synthetic_data['sub_label_nodes']
    sub_label_array = synthetic_data['sub_label']


def get_boundary(node_idx, dataset="syn1"):

    if dataset == "syn1":
        offset_idx = node_idx
    elif dataset == "syn2":
        offset_idx = node_idx
        # if node_idx < 700:
        #     offset_idx = node_idx - 400
        # else:
        #     offset_idx = node_idx - 800
    elif dataset == "syn3":
        offset_idx = node_idx
    elif dataset == "syn4":
        offset_idx = node_idx
    elif dataset == "syn8":
        offset_idx = node_idx



    boundary_list = []


    # rule_ix = rule_dict_node['idx2rule'][offset_idx]
    if isinstance(rule_dict_node['idx2rule'][offset_idx], list):
        rule_list = rule_dict_node['idx2rule'][offset_idx]
    else:
        rule_list = [rule_dict_node['idx2rule'][offset_idx]]
    for rule_ix in rule_list:
        rule = rule_dict_node['rules'][rule_ix]
        rule_label = rule['label']
        rule_imp_nodes = []
        rule_h_nodes = []
        h_nodes = []
        for b_num in range(len(rule['boundary'])):
            boundary = torch.from_numpy(rule['boundary'][b_num]['basis'])
            boundary = boundary.cuda()
            boundary_label = rule['boundary'][b_num]['label']
            boundary_list.append((boundary, boundary_label))
    return boundary_list



class ExplainerBoundaryInverse(explain.Explainer):
    def __init__(
        self,
        model,
        adj,
        feat,
        label,
        pred,
        train_idx,
        args,
        writer=None,
        print_training=True,
        graph_mode=False,
        graph_idx=False,
        num_nodes = None,
        rule_dict = None,
        device='cpu'
    ):
        super().__init__(model, adj, feat, label, pred, train_idx, args, writer, print_training, graph_mode, graph_idx, num_nodes, device)

        # self.model = model
        # self.model.eval()
        # self.adj = adj
        # self.feat = feat
        # self.label = label
        # self.pred = pred
        # self.train_idx = train_idx
        # self.num_nodes = num_nodes
        # self.n_hops = args.num_gc_layers
        # self.graph_mode = graph_mode
        # self.graph_idx = graph_idx
        # self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        # self.args = args
        # self.writer = writer
        # self.print_training = print_training


    # Main method
    def explain(
        self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model="exp", boundary = None
    ):
        global flips
        global inv_flips
        global incorrect_preds
        global pos_diff
        global inv_diff
        global topk_inv_diff
        global topk_inv_flips
        global total
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        sub_nodes = None
        if graph_mode:
            node_idx_new = node_idx   #set to 0, not used down the calls
            if len(self.adj.shape) < 3:
                sub_adj = self.adj
            else:
                sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            if self.num_nodes is not None:
                sub_nodes = self.num_nodes[graph_idx]
            else:
                sub_nodes = None
            neighbors = np.asarray(range(self.adj.shape[0])) #1,2,3....num_nodes
        else:
            print("node label: ", self.label[graph_idx][node_idx])
            if self.args.bmname == 'syn1' or self.args.bmname == 'syn2' or self.args.bmname == 'syn3' or self.args.bmname == 'syn4' or self.args.bmname == 'syn8':
                node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood_from_saved_data(
                    node_idx, self.args.bmname
                )
                boundary_list = get_boundary(node_idx, self.args.bmname)
                
            else:
                node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                    node_idx, graph_idx
                )


            print("neigh graph idx: ", node_idx, node_idx_new)
            sub_label = np.expand_dims(sub_label, axis=0)
        
        def shuffle_forward(l):
            order = list(range(len(l)))
            random.shuffle(order)
            return order

        def shuffle_backward(l):
            l_out = [0] * len(l)
            for i, j in enumerate(l):
                l_out[j] = l[i]
            return l_out

        order = list(range(sub_adj.shape[1]))
        rand_order = shuffle_forward(order)

        # sub_feat[rand_order, :] = sub_feat[order,:]
        # sub_adj[rand_order, :] = sub_adj[order,:]
        # sub_adj[:, rand_order] = sub_adj[:,order]
        
        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        print("xstart: ", torch.sum(x))
        label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            print("Graph predicted label: ", pred_label)
            if pred_label != label.item():
                incorrect_preds += 1
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            print("Node predicted label: ", pred_label[node_idx_new])

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
            sub_num_nodes=sub_nodes,
            device=self.device
        )
        if self.args.gpu:
            explainer = explainer.cuda()
            self.model = self.model.cuda()

        self.model.eval()
        if self.args.gpu:
            x = x.cuda()
            adj = adj.cuda()
            label = label.cuda()
        with torch.no_grad():
            if graph_mode:
               
                gt_preds, _ = self.model(x, adj, [sub_nodes.cpu().item()])

                gt_embedding, _ = self.model._getOutputOfOneLayer_Group(adj, x, [sub_nodes.cpu().item()])
                gt_preds = nn.Softmax(dim=0)(gt_preds[0])

            else:

                _, gt_embedding = self.model(x, adj, batch_num_nodes=None, new_node_idx=[node_idx_new])

        if not graph_mode:
            # use all boundaries
            boundary = [bound[0] for bound in boundary_list]
        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            print("training..............")
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                graph_embedding, ypred, inv_embedding, inv_pred = explainer(node_idx_new, unconstrained=unconstrained)

                loss = explainer.loss_boundary(ypred, pred_label, graph_embedding, boundary, gt_embedding.squeeze(0), inv_embedding, node_idx_new, epoch)
                # ypred = -1
                # loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                # with torch.no_grad():
                #     ypred = self.model()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()

                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred
                    )
                single_subgraph_label = sub_label.squeeze()

                if self.writer is not None:
                    self.writer.add_scalar("mask/density", mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    # if epoch % 25 == 0:
                    #     explainer.log_mask(epoch)
                    #     explainer.log_masked_adj(
                    #         node_idx_new, epoch, label=single_subgraph_label
                    #     )
                    #     explainer.log_adj_grad(
                    #         node_idx_new, pred_label, epoch, label=single_subgraph_label
                    #     )

                    if epoch == 0:
                        print("model.att: ", self.model.att)
                        if self.model.att:
                            # explain node
                            print("adj att size: ", adj_atts.size())
                            adj_att = torch.sum(adj_atts[0], dim=2)
                            # adj_att = adj_att[neighbors][:, neighbors]
                            node_adj_att = adj_att * adj.float()
                            io_utils.log_matrix(
                                self.writer, node_adj_att[0], "att/matrix", epoch
                            )
                            node_adj_att = node_adj_att[0].cpu().detach().numpy()
                            G = io_utils.denoise_graph(
                                node_adj_att,
                                node_idx_new,
                                threshold=3.8,  # threshold_num=20,
                                max_component=True,
                            )
                            io_utils.log_graph(
                                self.writer,
                                G,
                                name="att/graph",
                                identify_self=not self.graph_mode,
                                nodecolor="label",
                                edge_vmax=None,
                                args=self.args,
                            )
                    if model != "exp":
                        break

            print("finished training in ", time.time() - begin_time)
            if graph_mode:
                if torch.argmax(ypred).item() != pred_label:
                    flips += 1.0
            if model == "exp":
                masked_adj = (
                        explainer.masked_adj[0].cpu().detach().numpy()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()

            if graph_mode:
                with torch.no_grad():
                    final_preds = ypred
                    final_inv_preds = inv_pred
                    # final_preds,_ = self.model(x.cuda(), explainer.masked_adj, [sub_nodes.cpu().item()])
                    # final_preds = nn.Softmax(dim=0)(final_preds[0])

                    print("gt label: ", label.item())
                    print("gt preds: ", gt_preds)
                    print("final preds: ", final_preds)
                    gt_preds_np = gt_preds.cpu().numpy()

                    pos_diff[pred_label] += (gt_preds[pred_label] - final_preds[pred_label]).item()
                    if self.args.inverse_boundary_c > 0.:
                        inv_diff[pred_label] += (gt_preds[pred_label] - final_inv_preds[pred_label]).item()

                        if torch.argmax(final_inv_preds) == pred_label:
                            inv_flips[pred_label] += 1.0

                    topk_adj = noise_utils.filterTopK(masked_adj, sub_adj[0], k=4)
                    topk_adj_t = torch.from_numpy(topk_adj).float().cuda()
                    pred_topk, _ = self.model(x.cuda(), topk_adj_t.unsqueeze(0),
                                              batch_num_nodes=[sub_nodes.cpu().numpy()])

                    pred_topk = nn.Softmax(dim=0)(pred_topk[0])

                    topk_inv_diff[pred_label] += (gt_preds[pred_label] - pred_topk[pred_label]).item()

                    if torch.argmax(pred_topk) == pred_label:
                        topk_inv_flips[pred_label] += 1.0
                    total[pred_label] += 1.0
                    if self.args.draw_graphs:
                        accuracy_utils.saveAndDrawGraph(masked_adj, sub_adj[0], sub_feat[0],
                                                        self.num_nodes[graph_idx].item(),
                                                        self.args,
                                                        label.item(), pred_label, graph_idx,
                                                        prob=gt_preds[pred_label],
                                                        plt_path=None, adj_mask_bool=True)

                        accuracy_utils.saveAndDrawGraph(None, topk_adj, sub_feat[0], self.num_nodes[graph_idx].item(),
                                                        self.args,
                                                        label.item(), pred_label, graph_idx, prob=pred_topk[pred_label],
                                                        plt_path=None, adj_mask_bool=False)




            fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                    'node_idx_'+str(node_idx)+'graph_idx_'+str(graph_idx)+'.npy')
            if self.graph_mode:
                label_dir = os.path.join(self.args.logdir, ("label_" +str(pred_label)))
            else:
                label_dir = os.path.join(self.args.logdir, "node_explain")
            # with open(os.path.join(label_dir, fname), 'wb') as outfile:
                # np.save(outfile, np.asarray(masked_adj.copy()))
                # print("Saved adjacency matrix to ", fname)
        # masked_adj[order, :] = masked_adj[rand_order, :]
        # masked_adj[:, order] = masked_adj[:, rand_order]
        return masked_adj


    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        """
        Explain nodes

        Args:
            - node_indices  :  Indices of the nodes to be explained 
            - args          :  Program arguments (mainly for logging paths)
            - graph_idx     :  Index of the graph to explain the nodes from (if multiple).
        """
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices
        ]
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat, threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs


    def explain_nodes_gnn_stats(self, node_indices, graph_node_indices, args, graph_idx=0, model="exp"):

        global nbr_data
        global rule_dict_node

        log_name = self.args.prefix + "_logdir"
        log_path = os.path.join(self.args.ckptdir, log_name)
        if os.path.isdir(log_path):
            print("log dir already exists and will be overwritten")
            time.sleep(5)
        else:
            os.mkdir(log_path)

        log_file = self.args.prefix + "log_boundary_" + self.args.bmname + ".txt"
        log_file_path = os.path.join(log_path, log_file)
        myfile = open(log_file_path, "a")

        myfile.write("\n \n \n {}".format(self.args.bmname))
        myfile.write("\n method: {}".format(self.args.explainer_method))
        myfile.write("\n bloss version: {}".format(self.args.bloss_version))
        myfile.write("\n exp_path: {}".format(self.args.exp_path))

        myfile.write("\n lr: {}, bound cf: {}, size cf: {}, ent cf {}, lap cf {}, inv cf {}".format(self.args.lr, self.args.boundary_c,
                                                                              self.args.size_c, self.args.ent_c, self.args.lap_c, self.args.inverse_boundary_c))
        myfile.close()

        if self.args.bmname == "syn1":
            size = 700
            width = 4

        elif self.args.bmname == "syn2":
            size = 1400
            width = 10

        elif self.args.bmname == "syn3":
            size = 1020
            width = 10


        elif self.args.bmname == "syn4":
            size = 871
            width = 10

        elif self.args.bmname == "syn8":
            size = 660
            width = 5
        nbr_data = self.get_nbr_data(args, graph_node_indices, graph_idx)
        train_data, val_data = neighbor_utils.process_for_boundary(nbr_data, size, size, size, 100, width)

        rule_dict_node = extract.extract_rules(self.args.bmname, train_data, val_data, self.args, self.model.state_dict())

        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model)
            for node_idx in node_indices
        ]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        avg_map = 0.0
        map_d = {}
        count_d = {}
        AUC = accuracy_utils.AUC()
        avg_mask_density = 0.

        for i, idx in enumerate(node_indices):
            # new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood_from_saved_data(idx, self.args.bmname)
            # G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)

            pred = np.argmax(self.pred[graph_idx][nbrs], axis=1)
            labels = self.label[graph_idx][nbrs]
            # print(masked_adjs[i].shape, i, idx, new_idx)
            # print("nbrs: ", nbrs)
            # print("pred: ", pred)
            # print("label: ", labels)
            # print("n_adj\n", n_adj)
            # print("feat\n", feat)

            # if labels[new_idx] == 1:
            #     continue
            if self.args.bmname == 'syn3':

                map_score, h_edges = accuracy_utils.getmAPsyn3(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
            elif self.args.bmname == 'syn4':
                # map_score = getmAPsyn3(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])

                map_score, h_edges = accuracy_utils.getmAPsyn4(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
            elif self.args.bmname == 'syn1':
                # map_score = getmAPNodes(masked_adjs[i], n_adj, labels, nbrs, new_idx)

                map_score, h_edges = accuracy_utils.getmAPsyn1(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
            elif self.args.bmname == 'syn2':
                map_score, h_edges = accuracy_utils.getmAPsyn2(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
            else:
                map_score = accuracy_utils.getmAPNodes(masked_adjs[i], n_adj, labels, nbrs, new_idx)

            AUC.addEdgesFromDict(masked_adjs[i], h_edges)
            mask_density = np.sum(masked_adjs[i]) / np.sum(n_adj)
            avg_mask_density += mask_density

            if labels[new_idx] not in map_d:
                map_d[labels[new_idx]] = 0.
                count_d[labels[new_idx]] = 0.
            map_d[labels[new_idx]] += map_score
            count_d[labels[new_idx]] += 1.0

            avg_map += map_score



            print("map score: ", map_score,)

            continue

            # pred, real = self.make_pred_real(masked_adjs[i], new_idx)
            # pred_all.append(pred)
            # real_all.append(real)
            # denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
            # denoised_adj = nx.to_numpy_matrix(G)
            # graphs.append(G)
            # feats.append(denoised_feat)
            # adjs.append(denoised_adj)
            # io_utils.log_graph(
            #     self.writer,
            #     G,
            #     "graph/{}_{}_{}".format(self.args.dataset, model, i),
            #     identify_self=True,
            # )
        avg_mask_density = avg_mask_density/len(node_indices)

        myfile = open(log_file_path, "a")

        auc_res = AUC.getAUC()
        print(
            "ROC AUC score: {}".format(auc_res)
        )
        myfile.write("\n ROC AUC score: {}".format(auc_res))


        avg_map_score = avg_map / len(node_indices)
        myfile.write("\n mAP score: {}".format(avg_map_score))

        print(
            "Average mask density: {}".format(avg_mask_density)
        )
        myfile.write("\n Average mask density: {}".format(avg_mask_density))
        myfile.close()

        print("\n\navg map score: ", avg_map_score, "\n\n")
        for k in map_d.keys():
            print("label: ", k, "  map: ", map_d[k] / count_d[k], "  count: ", count_d[k])
        if args.fname != "":
            full_path = "./tuning/boundary/" + self.args.fname
            file1 = open(full_path, "a")  # write mode
            file1.write(str(args.size_c) + " " + str(args.lap_c) + " " + str(args.ent_c) + "\n")
            file1.write(str(avg_map_score) + "\n")
            for k in map_d.keys():
                file1.write(str(k) + " " +  str(map_d[k] / count_d[k]) + " " + str(count_d[k]) + " ")
            file1.write("\n\n")
            file1.close()

        return None

        pred_all = np.concatenate((pred_all), axis=0)
        real_all = np.concatenate((real_all), axis=0)

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_boundary_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_boundary_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        with open("log/pr/auc_boundary_" + self.args.dataset + "_" + model + ".txt", "w") as f:
            f.write(
                "dataset: {}, model: {}, auc: {}\n".format(
                    self.args.dataset, "exp", str(auc_all)
                )
            )

        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, args, graph_indices):

        if self.args.bmname == "synthetic" or self.args.bmname == "old_synthetic":
            load_sublabel_nodes(self.args)

        if self.args.apply_filter and self.args.bmname == 'Mutagenicity':
            # h_edges = accuracy_utils.gethedgesmutag()
            graph_indices, h_edges = accuracy_utils.filterMutag2(graph_indices, self.label, self.feat, self.adj,
                                                                 self.num_nodes)

        if args.draw_graphs:
            graph_indices = graph_indices[:5]
        # else:
        #     graph_indices = graph_indices[:300]

        """
        Explain graphs.
        """
        masked_adjs = []
        logging_graphs = False
        avg_top4_acc = 0.
        avg_top6_acc = 0.
        avg_top8_acc = 0.

        acc_count = 0.

        rule_top8_acc = 0.
        rule_acc_count = 0.
        ep_variance = 0.
        avg_mask_density = 0.
        mAP = 0.
        AUC = accuracy_utils.AUC()


        log_name = self.args.prefix + "_logdir"
        log_path = os.path.join(self.args.ckptdir, log_name)
        if os.path.isdir(log_path):
            print("log dir already exists and will be overwritten")
            time.sleep(5)
        else:
            os.mkdir(log_path)

        log_file = self.args.prefix + "log_boundary_inverse_" + self.args.bmname + ".txt"
        log_file_path = os.path.join(log_path, log_file)
        myfile = open(log_file_path, "a")

        myfile.write("\n \n \n {}".format(self.args.bmname))
        myfile.write("\n method: {}".format(self.args.explainer_method))
        myfile.write("\n bloss version: {}".format(self.args.bloss_version))
        myfile.write("\n exp_path: {}".format(self.args.exp_path))

        myfile.write("\n lr: {}, bound cf: {}, size cf: {}, ent cf {}, inv cf {}".format(self.args.lr, self.args.boundary_c,
                                                                              self.args.size_c, self.args.ent_c,
                                                                              self.args.inverse_boundary_c))
        myfile.close()

        if self.args.bmname == "synthetic" or self.args.bmname == "old_synthetic":
            size = 3000
            width = 12
        elif self.args.bmname == "Mutagenicity":
            size = 3000
            width = 14
        elif self.args.bmname == "NCI1":
            size = 3000
        elif self.args.bmname == "BA_2Motifs":
            size = 700
        train_data = (self.adj[:size], self.feat[:size], self.label[:size], self.num_nodes[:size])
        val_data = (self.adj[size-100:], self.feat[size-100:], self.label[size-100:], self.num_nodes[size-100:])
        rule_dict = extract.extract_rules(self.args.bmname, train_data, val_data, args, self.model.state_dict())

        # graph_indices = list(graph_indices)
        # random.shuffle(graph_indices)
        stats = accuracy_utils.Stats("GNNExplainer", self)

        for graph_idx in graph_indices:
            print("doing for graph index: ", graph_idx)

            #why node_idx is set to 0?
            print("denoising...")


               
            rule_ix = rule_dict['idx2rule'][graph_idx]
            rule = rule_dict['rules'][rule_ix]
            rule_label = rule['label']
            rule_imp_nodes = []
            rule_h_nodes = []
            boundary_list = []
            h_nodes = []
            for b_num in range(len(rule['boundary'])):

                boundary = torch.from_numpy(rule['boundary'][b_num]['basis'])
                if self.args.gpu:
                    boundary = boundary.cuda()
                boundary_label = rule['boundary'][b_num]['label']
                boundary_list.append(boundary)
                # h_nodes.extend(explain.getHnodes(graph_idx, b_num))


                # masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True, boundary = boundary)
            masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True, boundary = boundary_list)
            print("flips: ", flips)
            sub_adj_np = self.adj[graph_idx].cpu().numpy()
            mask_density = np.sum(masked_adj)/np.sum(sub_adj_np)

            variance = np.sum(np.abs(masked_adj - 0.5) * sub_adj_np) / np.sum(sub_adj_np)
            ep_variance += variance
            label = self.label[graph_idx]

            if self.args.bmname == 'synthetic' or self.args.bmname == 'old_synthetic':

                h_nodes = accuracy_utils.getHNodes(graph_idx, sub_label_nodes, sub_label_array, self.args)
                ht_edges = accuracy_utils.getHTEdges(h_nodes, sub_adj_np)
                AUC.addEdges2(masked_adj, ht_edges)

                if self.args.draw_graphs:

                    gt_mask = sub_adj_np - 0.9
                    for e in ht_edges.keys():
                        gt_mask[e[0], e[1]] = 1.0
                        gt_mask[e[1], e[0]] = 1.0

                    # gt_mask
                    accuracy_utils.saveAndDrawGraph(gt_mask, sub_adj_np, self.feat[graph_idx].cpu().numpy(),
                                                    self.num_nodes[graph_idx].item(),
                                                    self.args,
                                                    label.item(), label.item(), graph_idx,
                                                    prob=1.0,
                                                    plt_path=None, adj_mask_bool=True, prefix="gt_")

                # AUC.addEdges(masked_adj, h_nodes, dataset='synthetic')

                mAP_s = accuracy_utils.getmAP(masked_adj, h_nodes)
                mAP += mAP_s
                thresh_nodes = 8
                imp_nodes = explain.getImportantNodes(masked_adj, 8)
                top4_acc, top6_acc, top8_acc = accuracy_utils.getAcc(imp_nodes, h_nodes)

                rule_imp_nodes = imp_nodes
                rule_h_nodes = h_nodes
                avg_top4_acc += top4_acc
                avg_top6_acc += top6_acc
                avg_top8_acc += top8_acc
                print("rule top4 acc: {}, top6 acc: {}, top8 acc: {}".format(top4_acc, top6_acc, top8_acc))


            else:
                rule_imp_nodes = explain.getImportantNodes(masked_adj, 8)

                thresh_nodes = 15
                # rule_imp_nodes = None
                rule_h_nodes = None

            if self.args.bmname == 'Mutagenicity' and self.args.apply_filter:
                ht_edges = h_edges[graph_idx]
                AUC.addEdges2(masked_adj, ht_edges)

            print("Done......")
            print("Graph number: {}, Rule number: {}, Rule label: {}".format(
                graph_idx,
                rule_ix,
                rule_label))


            avg_mask_density += mask_density
            acc_count += 1.0


            masked_adjs.append(masked_adj)


            stats.update(masked_adj, rule_imp_nodes, graph_idx)

            if rule_h_nodes is not None:
                top4_acc, top6_acc, top8_acc = accuracy_utils.getAcc(rule_imp_nodes, rule_h_nodes)
                rule_top8_acc += top8_acc
            rule_acc_count += 1.0


        myfile = open(log_file_path, "a")
        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")
        if self.args.bmname =='synthetic' or self.args.bmname =='old_synthetic':
            print(
                "Boundary wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(avg_top4_acc / acc_count, avg_top6_acc / acc_count, avg_top8_acc/acc_count)
            )

            myfile.write("\n Boundary wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(avg_top4_acc / acc_count, avg_top6_acc / acc_count, avg_top8_acc/acc_count))

            print(
                "Rule wise top8 acc: {}".format(rule_top8_acc / rule_acc_count)
            )
            print(
                "mAP score: {}".format(mAP / rule_acc_count)
            )
            myfile.write("\n mAP score: {}".format(mAP / rule_acc_count))

            print(
                "ROC AUC score: {}".format(AUC.getAUC())
            )
            myfile.write("\n ROC AUC score: {}".format(AUC.getAUC()))


        if self.args.bmname == 'Mutagenicity' and self.args.apply_filter:
            print(
                "ROC AUC score: {}".format(AUC.getAUC())
            )
            myfile.write("\n ROC AUC score: {}".format(AUC.getAUC()))


        print(
            "Average mask density: {}".format(avg_mask_density / acc_count)

        )
        total[total<0.5] = 1.0

        myfile.write("\n Average mask density: {}".format(avg_mask_density / rule_acc_count))
        print(
            "pos diff: {}, inv diff: {}, k: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total, self.args.topk,
                                                                          topk_inv_diff / total)

        )

        myfile.write("\n pos diff: {}, inv diff: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total,
                                                                               topk_inv_diff / total))

        print("Variance: ", ep_variance / rule_acc_count)
        myfile.write("\n Variance: {}".format(ep_variance / rule_acc_count))

        print("Flips: ", flips)
        print("inv Flips: ", inv_flips)
        print("topk inv Flips: ", topk_inv_flips)
        print("Incorrect preds: ", incorrect_preds)
        print("Total: ", total)

        myfile.write(
            "\n flips: {}, Inv flips: {}, topk: {}, topk Inv flips: {}, Incorrect preds: {}, Total: {}".format(flips,
                                                                                                               inv_flips,
                                                                                                               self.args.topk,
                                                                                                               topk_inv_flips,
                                                                                                               incorrect_preds,
                                                                                                               total))
        print(stats)
        myfile.write(str(stats))

        myfile.close()
        return masked_adjs

    

class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        args,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
        sub_num_nodes = None,
        device='cpu'
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.sub_num_nodes = sub_num_nodes
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode
        self.device = 'cuda:0'
        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)


        self.coeffs = {
            # "size": 0.04, #mutag
            "size":1.8,#synth
            # "size": 0.03,  # syn1
            #"size": 0.009,  # syn2
            #"size": 0.03,#0.08, 0.005,0.2 and 0.3 for synthetic
            "feat_size": 0.0,#1.0
            "ent": 1.0, #1.0,
            "feat_ent": 0.0,
            "grad": 0,
            "lap": 0.0, #1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes)).to(self.device)
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _inverse_masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = 1.0- sym_mask
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        inv_masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            inv_masked_adj += (bias + bias.t()) / 2
        return inv_masked_adj * self.diag_mask

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx, unconstrained=False, mask_features=True, marginalize=False):
        print(self.device)
        x = self.x.to(self.device)

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask #making it symmetric,
                # only true for undirected graphs?
            )
        else: #applicable for graph explanation
            self.masked_adj = self._masked_adj()
            self.inverse_masked_adj = self._inverse_masked_adj()

            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize: # add noise or mask features
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x
                    # x = x * feat_mask

        if self.masked_adj.shape[1] == 5129:
            ypred = self.model.predict(x, probs=False, adj=self.masked_adj.squeeze(0))
            adj_att = None
        # else:
        # graph_embedding, = self.model._getOutputOfOneLayer_Group(self.masked_adj,x, batch_num_nodes=[self.sub_num_nodes.cpu().numpy()])
        if self.sub_num_nodes is not None:
            sub_num_nodes_l = [self.sub_num_nodes.cpu().numpy()]
        else:
            sub_num_nodes_l = None

        graph_embedding = None
        ypred = None
        inv_embedding = None
        inv_pred = None
        if self.graph_mode:
            # if self.args.boundary_c > 0.0:
            ypred, graph_embedding = self.model(x, self.masked_adj, batch_num_nodes=sub_num_nodes_l)
            ypred = nn.Softmax(dim=0)(ypred[0])
            graph_embedding = graph_embedding.squeeze(0)
            if self.args.inverse_boundary_c > 0.0:
                inv_pred, inv_embedding = self.model(x, self.inverse_masked_adj, batch_num_nodes=sub_num_nodes_l)
                inv_pred = nn.Softmax(dim=0)(inv_pred[0])
                inv_embedding = inv_embedding.squeeze(0)

            return graph_embedding, ypred, inv_embedding, inv_pred

        else:

            ypred, graph_embedding = self.model(x, self.masked_adj, batch_num_nodes=sub_num_nodes_l, new_node_idx=[node_idx])
            node_pred = ypred[self.graph_idx, node_idx, :]
            node_pred = nn.Softmax(dim=0)(node_pred)
            return graph_embedding, node_pred
        return res, adj_att

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        if adj.shape[1] == 5129:
            if len(adj.shape) == 3:
                adj_sq = adj.squeeze(0)
            ypred = self.model.predict(x, probs=False, adj=adj_sq)
        else:
            ypred, _ = self.model(x, adj)

        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss_boundary(self, pred, pred_label, graph_embedding, boundary_list, gt_embedding, inv_embedding, node_idx, epoch):
        # print(self.args.mask_bias, pred, pred_label)
        # exit()
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        pred_loss = torch.zeros(1).cuda()

        pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
        # gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
        logit = pred[pred_label_node]

        # logit = pred[gt_label_node]
        pred_loss = -torch.log(logit)



        if self.args.bloss_version == 'proj':
            boundary_loss = 0.
            for boundary in boundary_list:
                boundary_loss += torch.norm(torch.sum((graph_embedding - gt_embedding) * boundary[:20]))
            boundary_loss = (boundary_loss / len(boundary_list))  # 0.3
            net_boundary_loss = boundary_loss
        elif self.args.bloss_version == 'sigmoid':
            if self.args.boundary_c < 0.0:
                boundary_loss = torch.zeros(1).cuda()
            else:
                boundary_loss = 0.
                sigma = 1.0
                for boundary in boundary_list:
                    gt_proj = torch.sum(gt_embedding * boundary[:20]) + boundary[20]
                    ft_proj = torch.sum(graph_embedding * boundary[:20]) + boundary[20]
                    boundary_loss += torch.nn.functional.sigmoid(-1.0 * sigma * (gt_proj * ft_proj))
                boundary_loss = self.args.boundary_c * (boundary_loss / len(boundary_list))

            if self.args.inverse_boundary_c < 0.0:
                net_boundary_loss = boundary_loss
            else:
                sigma = 1.0
                inv_losses = []
                for boundary in boundary_list:
                    gt_proj = torch.sum(gt_embedding * boundary[:20]) + boundary[20]
                    inv_proj = torch.sum(inv_embedding * boundary[:20]) + boundary[20]
                    # print("inv: ", gt_proj, inv_proj)
                    inv_loss = torch.nn.functional.sigmoid(sigma * (gt_proj * inv_proj))
                    inv_losses.append(inv_loss)

                inv_losses_t = torch.stack(inv_losses)
                # print("debug: ", inv_losses, torch.min(inv_losses_t))

                inverse_boundary_loss = self.args.inverse_boundary_c * torch.min(inv_losses_t)

                net_boundary_loss = boundary_loss + inverse_boundary_loss
                print("Boundary loss: {}, Inverse boundary loss: {}".format(boundary_loss.item(),
                                                                            inverse_boundary_loss.item()))



        else:
            assert (False)

        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)


        if self.args.size_c > -0.001:
            size_loss = self.args.size_c * torch.sum(mask)
        else:
            size_loss = self.coeffs["size"] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy for making it discrete
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        if self.args.ent_c > -0.001:
            mask_ent_loss = self.args.ent_c * torch.mean(mask_ent)
        else:
            mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask             \
                        * torch.log(feat_mask)  \
                        - (1 - feat_mask)       \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = torch.zeros((1)).cuda()
        else:
            if self.args.lap_c > -0.001:
                lap_loss = (self.args.lap_c
                            * (pred_label_t @ L @ pred_label_t)
                            / self.adj.numel()
                            )
            else:
                lap_loss = (self.coeffs["lap"]
                    * (pred_label_t @ L @ pred_label_t)
                    / self.adj.numel()
                )


        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)
        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss

        # loss = net_boundary_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        print("net boundary loss: {}, pred loss {},  size loss: {}, lap loss: {}, mask ent loss: {}, feat size loss: {} ".format(net_boundary_loss.item(),
                                                                                                              pred_loss.item(),
                                                                                                              size_loss.item(),
                                                                                                              lap_loss.item(),
                                                                                                              mask_ent_loss.item(),
                                                                                                              feat_size_loss.item()))
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", boundary_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

    
