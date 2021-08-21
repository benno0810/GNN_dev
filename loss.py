from model import *
from utils import *

class ModularityScore(th.nn.Module):
    def __init__(self,n_classes,cuda,direction,Q):
        super(ModularityScore, self).__init__()
        ## define C as parameter
        #self.params = th.nn.ParameterList([C])
        self.cuda=cuda
        self.direction=direction
        self.Q = Q


    def forward(self,C):
        # -tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(C),Q),C))
        #C=th.sigmoid(C)
        Q=self.Q.float()
        if self.cuda:
            C=C.cuda()
            Q=Q.cuda()
        temp = th.matmul(th.matmul(C.t(), Q), C)
        if self.direction ==-1:
            loss = -temp.trace()
        elif self.direction==1:
                loss = temp.trace()

        return loss


class CrossEntropyLoss(th.nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss
