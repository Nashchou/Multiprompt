import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import process
from process import process_tu

dataset = 'pubmed'

# training params
batch_size = 300
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = False
nonlinearity = 'prelu'  # special name to separate parameters

i = 0
class_num = 2

node_class_num = 3

dataset = TUDataset(root='data', name='BZR', use_node_attr=True)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

shotnumlist = [40,50]
# random.seed(39)
for step, data in enumerate(loader):
    if step == 0:
        for i in range(0, 100):
            cnt = 0
            cnt0 = 0
            cnt1 = 0
            cnt2 = 0
            cnt3 = 0
            cnt4 = 0
            cnt5 = 0
            cnt6 = 0
            for shotnum in shotnumlist:
                if shotnum == shotnumlist[0]:
                    graphlen = [0] * (shotnum * class_num)
                    graph_list =  [0] * (shotnum * class_num)
                    # graph_labels = torch.Tensor(shotnum * class_num)
                else:
                    graphlen = pastlen + [0] *  class_num * (shotnum - pastshotnum)
                    graph_list = pastlist + [0] *  class_num * (shotnum - pastshotnum)
                    # graph_labels = torch.Tensor(shotnum * class_num)

                # os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}".format(shotnum,i))

                # for n1 in range(0, 3):
                #     for n2 in range(0, 3):
                #         trainfeature[n1][n2] = 0
                for m in range(0, 10000):

                    j = random.randint(0, 199)
                    trainfeature = data[j].x
                    # print(type(trainfeature))
                    trainfeature = trainfeature[:, node_class_num]

                    if cnt == class_num * shotnum:
                        pastshotnum = shotnum
                        pastlist = graph_list
                        pastlen = graphlen
                        if not os.path.isdir("data/fewshot_BZR_graph/{}shot_BZR_graph/{}".format(shotnum, i)):
                            os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}".format(shotnum, i))
                        else:
                            break
                        minidataset = data[graph_list]

                        feature, adj, lbls, lens = process_tu(minidataset, node_class_num, shotnum * class_num)

                        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
                        adj = (adj + sp.eye(adj.shape[0])).todense()

                        adj = torch.FloatTensor(adj)
                        feature = torch.FloatTensor(feature)
                        graphlen = torch.Tensor(graphlen)
                        torch.save(feature,
                                   "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/feature.pt".format(shotnum,
                                                                                                    i))
                        torch.save(adj,
                                   "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/adj.pt".format(shotnum, i))
                        torch.save(lbls,
                                   "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/labels.pt".format(shotnum, i))
                        torch.save(lens,
                                   "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/graph_len.pt".format(shotnum,
                                                                                                      i))
                        # # os.removedir("fewshot/{}".format(i))
                        #
                        # torch.save(trainlal, "fewshot_BZR/{}/nodelabels.pt".format(i))
                        #
                        # torch.save(trainidx, "fewshot_BZR/{}/nodeidx.pt".format(i))
                        # for n1 in range(0, 3):
                        #     for n2 in range(0, 3):
                        #         trainadj[n1][n2] = adj[0][mark[n1]][mark[n2]]
                        #
                        # print('number', i, 'trainlal', trainlal, '\n')
                        #
                        # print('trainfeature', trainfeature, '\n')
                        #
                        # print("adj", trainadj)


                        break

                    if data[j].y[0].item() == 0 and cnt0 != shotnum:
                        graphlen[cnt] = trainfeature.shape[0]
                        graph_list[cnt] = j
                        # graph_labels[cnt] = 0
                        # trainlal[0][cnt] = 0
                        # os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(shotnum,i, cnt))
                        # torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(shotnum,i, cnt))
                        # # trainidx[cnt] = j
                        # # list[j] = 1
                        # torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(shotnum,i, cnt))
                        # torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(shotnum,i, cnt))
                        # torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(shotnum,i, cnt))
                        print(j, ',0')
                        cnt = cnt + 1
                        cnt0 += 1

                    if data[j].y[0].item() == 1 and cnt1 != shotnum:
                        graphlen[cnt] = trainfeature.shape[0]
                        graph_list[cnt] = j
                        # graph_labels[cnt] = 1
                        # os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(shotnum,i, cnt))
                        # torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(shotnum,i, cnt))
                        # # trainidx[cnt] = j
                        # # list[j] = 1
                        # torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(shotnum,i, cnt))
                        # torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(shotnum,i, cnt))
                        # torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(shotnum,i, cnt))
                        print(j, ',1')
                        # mark[cnt] = j
                        cnt = cnt + 1
                        cnt1 += 1
                    if data[j].y[0].item() == 2 and cnt2 != shotnum:
                        graphlen[cnt] = trainfeature.shape[0]
                        graph_list[cnt] = j
                        # graph_labels[cnt] = 2
                        # os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(shotnum,i, cnt))
                        # torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(shotnum,i, cnt))
                        # # trainidx[cnt] = j
                        # # list[j] = 1
                        # torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(shotnum,i, cnt))
                        # torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(shotnum,i, cnt))
                        # torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(shotnum,i, cnt))
                        print(j, ',2')
                        # mark[cnt] = j
                        cnt = cnt + 1
                        cnt2 += 1
                    if data[j].y[0].item() == 3 and cnt3 != shotnum:
                        graphlen[cnt] = trainfeature.shape[0]
                        graph_list[cnt] = j
                        # graph_labels[cnt] = 3
                        # os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(shotnum,i, cnt))
                        # torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(shotnum,i, cnt))
                        # # trainidx[cnt] = j
                        # # list[j] = 1
                        # torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(shotnum,i, cnt))
                        # torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(shotnum,i, cnt))
                        # torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(shotnum,i, cnt))
                        print(j, ',3')
                        # mark[cnt] = j
                        cnt = cnt + 1
                        cnt3 += 1
                    if data[j].y[0].item() == 4 and cnt4 != shotnum:
                        graphlen[cnt] = trainfeature.shape[0]
                        graph_list[cnt] = j
                        # graph_labels[cnt] = 4
                        # os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(shotnum,i, cnt))
                        # torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(shotnum,i, cnt))
                        # # trainidx[cnt] = j
                        # # list[j] = 1
                        # torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(shotnum,i, cnt))
                        # torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(shotnum,i, cnt))
                        # torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(shotnum,i, cnt))
                        print(j, ',4')
                        # mark[cnt] = j
                        cnt = cnt + 1
                        cnt4 += 1
                    if data[j].y[0].item() == 5 and cnt5 != shotnum:
                        graphlen[cnt] = trainfeature.shape[0]
                        graph_list[cnt] = j
                        # graph_labels[cnt] = 5
                        # os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(shotnum,i, cnt))
                        # torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(shotnum,i, cnt))
                        # # trainidx[cnt] = j
                        # # list[j] = 1
                        # torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(shotnum,i, cnt))
                        # torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(shotnum,i, cnt))
                        # torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(shotnum,i, cnt))
                        print(j, ',5')
                        # mark[cnt] = j
                        cnt = cnt + 1
                        cnt5 += 1
                    # if data[j].y[0].item() == 2 and cnt2 != shotnum:
                    #     os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(i, cnt))
                    #     torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(i, cnt))
                    #     # trainidx[cnt] = j
                    #     # list[j] = 1
                    #     torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(i, cnt))
                    #     torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(i, cnt))
                    #     torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(i, cnt))
                    #     print(j, ',2')
                    #     # mark[cnt] = j
                    #     cnt = cnt + 1
                    #     cnt2 += 1
                    # if data[j].y[0].item() == 3 and cnt3 != shotnum:
                    #     os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(i, cnt))
                    #     torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(i, cnt))
                    #     # trainidx[cnt] = j
                    #     # list[j] = 1
                    #     torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(i, cnt))
                    #     torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(i, cnt))
                    #     torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(i, cnt))
                    #     # mark[cnt] = j
                    #     cnt = cnt + 1
                    #     cnt3 += 1
                    # if data[j].y[0].item() == 4 and cnt4 != shotnum:
                    #     os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(i, cnt))
                    #     torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(i, cnt))
                    #     # trainidx[cnt] = j
                    #     # list[j] = 1
                    #     torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(i, cnt))
                    #     torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(i, cnt))
                    #     torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(i, cnt))
                    #     print(j, ',4')
                    #     # mark[cnt] = j
                    #     cnt = cnt + 1
                    #     cnt4 += 1
                    # if data[j].y[0].item() == 5 and cnt5 != shotnum:
                    #     os.makedirs("data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}".format(i, cnt))
                    #     torch.save(e_ind, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/e_ind.pt".format(i, cnt))
                    #     # trainidx[cnt] = j
                    #     # list[j] = 1
                    #     torch.save(trainfeature, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphemb.pt".format(i, cnt))
                    #     torch.save(trainadj, "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphadj.pt".format(i, cnt))
                    #     torch.save(data[j].y[0], "data/fewshot_BZR_graph/{}shot_BZR_graph/{}/{}/graphlabel.pt".format(i, cnt))
                    #     print(j, ',5')
                    #     # mark[cnt] = j
                    #     cnt = cnt + 1
                    #     cnt5 += 1
                    # if lals[0][j].item() == 6 and cnt6 == 0:
                    #     trainlal[0][cnt] = 6
                    #     trainfeature[cnt][2] = 1
                    #     # list[j] = 1
                    #     # print(j, ' ,')
                    #     cnt = cnt + 1
                    #     cnt6 = 1


        break
    else:
        os.makedirs("data/fewshot_BZR_graph/testset")
        testset = data[range(100)]
    
        feature, adj, labels, graphlen = process_tu(testset, node_class_num, 100)
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = (adj + sp.eye(adj.shape[0])).todense()
    
        adj = torch.FloatTensor(adj)
        feature = torch.FloatTensor(feature)
        graphlen = torch.Tensor(graphlen)
        torch.save(feature, "data/fewshot_BZR_graph/testset/feature.pt")
        torch.save(adj,
                   "data/fewshot_BZR_graph/testset/adj.pt")
        torch.save(labels,
                   "data/fewshot_BZR_graph/testset/labels.pt")
        torch.save(graphlen,
                   "data/fewshot_BZR_graph/testset/graph_len.pt")
   
print("end")
