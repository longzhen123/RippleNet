import time
import numpy as np
import torch.nn as nn
import torch as t
from sklearn.metrics import accuracy_score, roc_auc_score
from src.evaluate import get_all_metrics
from src.load_base import load_data, get_records


class RippleNet(nn.Module):

    def __init__(self, dim, n_entities, H, n_rel, l1, l2):
        super(RippleNet, self).__init__()

        self.dim = dim
        self.H = H
        self.l1 = l1
        self.l2 = l2
        ent_emb = t.randn(n_entities, dim)
        rel_emb = t.randn(n_rel, dim, dim)
        nn.init.xavier_uniform_(ent_emb)
        nn.init.xavier_uniform_(rel_emb)
        self.ent_emb = nn.Parameter(ent_emb)
        self.rel_emb = nn.Parameter(rel_emb)
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, pairs, ripple_sets):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        item_embeddings = self.ent_emb[items]
        heads_list, relations_list, tails_list = self.get_head_relation_and_tail(users, ripple_sets)
        user_represents = self.get_vector(items, heads_list, relations_list, tails_list)

        predicts = t.sigmoid((user_represents * item_embeddings).sum(dim=1))

        return predicts

    def get_head_relation_and_tail(self, users, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []
        for h in range(self.H):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for user in users:

                l_head_list.extend(ripple_sets[user][h][0])
                l_relation_list.extend(ripple_sets[user][h][1])
                l_tail_list.extend(ripple_sets[user][h][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)

        return heads_list, relations_list, tails_list

    def get_vector(self, items, heads_list, relations_list, tails_list):

        o_list = []
        item_embeddings = self.ent_emb[items].view(-1, self.dim, 1)
        for h in range(self.H):
            head_embeddings = self.ent_emb[heads_list[h]].view(len(items), -1, self.dim, 1)
            relation_embeddings = self.rel_emb[relations_list[h]].view(len(items), -1, self.dim, self.dim)
            tail_embeddings = self.ent_emb[tails_list[h]].view(len(items), -1, self.dim)

            Rh = t.matmul(relation_embeddings, head_embeddings).view(len(items), -1, self.dim)
            hRv = t.matmul(Rh, item_embeddings)
            pi = t.softmax(hRv, dim=1)
            o_embeddings = (pi * tail_embeddings).sum(dim=1)
            o_list.append(o_embeddings)

        return sum(o_list)

    def computer_loss(self, labels, predicts, users, ripple_sets):

        base_loss = self.criterion(predicts, labels)
        kg_loss = 0
        for h in range(self.H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for user in users:
                h_head_list.extend(ripple_sets[user][h][0])
                h_relation_list.extend(ripple_sets[user][h][1])
                h_tail_list.extend(ripple_sets[user][h][2])

            head_emb = self.ent_emb[h_head_list].view(-1, 1, self.dim)  # (n, dim)-->(n, 1, dim)
            rel_emb = self.rel_emb[h_relation_list].view(-1, self.dim, self.dim)  # (n, dim, dim)
            tail_emb = self.ent_emb[h_relation_list].view(-1, self.dim, 1)  # (n, dim)-->(n, dim, 1)

            Rt = t.matmul(rel_emb, tail_emb)  # (n, dim, 1)
            hRt = t.matmul(head_emb, Rt)  # (n, 1, 1)

            kg_loss = kg_loss - t.sigmoid(hRt).mean()

        return base_loss + self.l1 * kg_loss


def get_scores(model, rec, ripple_sets):
    scores = {}
    model.eval()
    for user in (rec):

        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = model.forward(pairs, ripple_sets).cpu().view(-1).detach().numpy().tolist()
        # print(predict)
        n = len(pairs)
        user_scores = {items[i]: predict[i] for i in range(n)}
        user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        scores[user] = user_list
    model.train()
    return scores


def eval_ctr(model, pairs, ripple_sets, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model(pairs[i: i+batch_size], ripple_sets).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return round(auc, 3), round(acc, 3)


def get_ripple_set(train_dict, kg_dict, H, size):

    ripple_set_dict = {user: [] for user in train_dict}

    for u in (train_dict):

        next_e_list = train_dict[u]

        for h in range(H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for head in next_e_list:
                if head not in kg_dict:
                    continue
                for rt in kg_dict[head]:
                    relation = rt[0]
                    tail = rt[1]
                    h_head_list.append(head)
                    h_relation_list.append(relation)
                    h_tail_list.append(tail)

            if len(h_head_list) == 0:
                h_head_list = ripple_set_dict[u][-1][0]
                h_relation_list = ripple_set_dict[u][-1][1]
                h_tail_list = ripple_set_dict[u][-1][0]
            else:
                replace = len(h_head_list) < size
                indices = np.random.choice(len(h_head_list), size, replace=replace)
                h_head_list = [h_head_list[i] for i in indices]
                h_relation_list = [h_relation_list[i] for i in indices]
                h_tail_list = [h_tail_list[i] for i in indices]

            ripple_set_dict[u].append((h_head_list, h_relation_list, h_tail_list))

            next_e_list = ripple_set_dict[u][-1][2]

    return ripple_set_dict


def train(args, is_topk=False):
    np.random.seed(555)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    train_records = get_records(train_set)
    test_records = get_records(test_set)
    ripple_sets = get_ripple_set(train_records, kg_dict, args.H, args.K_h)

    model = RippleNet(args.dim, n_entity, args.H, n_relation, args.l1, args.l2)
    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('H: %d' % args.H, end='\t')
    print('K_h: %d' % args.K_h, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    all_precision_list = []
    for epoch in (range(args.epochs)):

        start = time.clock()
        loss_sum = 0
        np.random.shuffle(train_set)
        for i in range(0, len(train_set), args.batch_size):

            if (i + args.batch_size + 1) > len(train_set):
                batch_uvls = train_set[i:]
            else:
                batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)

            users = [pair[0] for pair in pairs]
            predicts = model(pairs, ripple_sets)

            loss = model.computer_loss(labels, predicts, users, ripple_sets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_auc, train_acc = eval_ctr(model, train_set, ripple_sets, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, ripple_sets, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, ripple_sets, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec, ripple_sets)
            precision_list = get_all_metrics(scores, test_records)[0]
            print(precision_list, end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        all_precision_list.append(precision_list)
        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.3f \t train_acc: %.3f \t eval_auc: %.3f \t eval_acc: %.3f \t '
          'test_auc: %.3f \t test_acc: %.3f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print(all_precision_list[indices])

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]
