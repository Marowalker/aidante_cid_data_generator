import networkx as nx
import re
import models
from feature_engineering.deptree.deptree_model import DepTree
from module import plotlib
import numpy as np
import constants
from utils import load_vocab, get_trimmed_w2v_vectors


class Finder:
    def __init__(self):
        pass

    def normalize_deptree(self, deptree):
        edges = []
        norm_tree = deptree.copy()

    @staticmethod
    def parse_directed_sdp(sdp):
        if len(sdp) != 0:
            return ' '.join([elem.content + '\\' + elem.metadata['pos_tag'] + '\\' + str(elem.metadata['hypernym']) if
                             issubclass(type(elem), models.Token) else '(' + elem[
                1] + '_' + (elem[0] if '||' not in elem[0] else elem[0].split('||', 1)[0]) + ')' for elem in sdp])
        else:
            return None

    @staticmethod
    def find_sdp_with_word_only(deptree, from_token, to_token):
        """
        :param deptree: [(rel, parent, child)]
        :param from_token: token-index
        :param to_token:  token-index
        :return string: sdp from from_token to to_token
        """
        edges = []
        for rel, pa, ch in deptree:
            edges.append((pa, ch))
        graph = nx.Graph(edges)
        path = nx.shortest_path(graph, from_token, to_token)

        return ' '.join([re.sub('\d+', '', node)[:-1] for node in path])

    @staticmethod
    def find_sdp_with_relation(deptree, from_token, to_token):
        """
        :param deptree: [(rel, parent, child)]
        :param from_token: token-index
        :param to_token:  token-index
        :return string: sdp from from_token to to_token
        """
        edges = []
        rel_map = {}
        for rel, pa, ch in deptree:
            edges.append((pa, ch))
            rel_map[(pa, ch)] = rel

        graph = nx.Graph(edges)

        path = nx.shortest_path(graph, from_token, to_token)
        final_path = ""
        for i in range(len(path)):
            node = re.sub('\d+', '', path[i])[:-1]

            if i == len(path) - 1:
                final_path += node
            else:
                if not (path[i], path[i + 1]) in edges:
                    rela = rel_map[(path[i + 1], path[i])]
                else:
                    rela = rel_map[(path[i], path[i + 1])]
                final_path += node + ' (' + rela + ') '
        return final_path

    @staticmethod
    def find_sdp_with_directed_relation(deptree, from_token, to_token):
        """
        :param deptree:
        :param from_token: token-index
        :param to_token:  token-index
        :return string: sdp from from_token to to_token
        """
        edges = []
        rel_map = {}
        for rel, pa, ch in deptree:
            edges.append((pa, ch))
            rel_map[(pa, ch)] = rel

        graph = nx.Graph(edges)
        final_path = ""
        try:
            path = nx.shortest_path(graph, from_token, to_token)
            for i in range(len(path)):
                node = re.sub('\d+', '', path[i])[:-1]

                if i == len(path) - 1:
                    final_path += node
                else:
                    if not (path[i], path[i + 1]) in edges:
                        rela = 'r_' + rel_map[(path[i + 1], path[i])]
                    else:
                        rela = 'l_' + rel_map[(path[i], path[i + 1])]
                    final_path += node + ' (' + rela + ') '
        except Exception:
            print(edges)
            print(deptree)
            print(from_token)
            print(to_token)

        return final_path

    @staticmethod
    def find_children(deptree, node):
        res = []
        for rel, pa, ch in deptree:
            if pa.content + '-' + str(pa.sent_offset[0]) == node:
                child = ch.content.lower()
                res.append(child)
        return res

    @staticmethod
    def find_sdp(deptree, from_token, to_token):
        edges = []
        rel_map = {}
        token_map = {}
        for rel, pa, ch in deptree:
            fro = pa.content + '-' + str(pa.sent_offset[0])
            to = ch.content + '-' + str(ch.sent_offset[0])
            edges.append((fro, to))
            rel_map[(fro, to)] = rel
            token_map[fro] = pa
            token_map[to] = ch

        graph = nx.Graph(edges)
        final_path = []
        try:
            path = nx.shortest_path(graph, from_token.content + '-' + str(from_token.sent_offset[0]),
                                    to_token.content + '-' + str(to_token.sent_offset[0]))
            for i in range(len(path)):
                if i == len(path) - 1:
                    final_path += [token_map[path[i]]]
                else:
                    if not (path[i], path[i + 1]) in edges:
                        rela = (rel_map[(path[i + 1], path[i])], 'r')
                    else:
                        rela = (rel_map[(path[i], path[i + 1])], 'l')
                    final_path += [token_map[path[i]]]
                    final_path.append(rela)

        except Exception as e:
            print('Error')
        return final_path

    @staticmethod
    def find_sdp_with_sibling(deptree, from_token, to_token):
        edges = []
        rel_map = {}
        token_map = {}
        for rel, pa, ch in deptree:
            fro = pa.content + '-' + str(pa.sent_offset[0])
            to = ch.content + '-' + str(ch.sent_offset[0])
            edges.append((fro, to))
            rel_map[(fro, to)] = rel
            token_map[fro] = pa
            token_map[to] = ch

        graph = nx.Graph(edges)
        final_path = []
        sb_path = []
        try:
            path = nx.shortest_path(graph, from_token.content + '-' + str(from_token.sent_offset[0]),
                                    to_token.content + '-' + str(to_token.sent_offset[0]))
            # print(path)
            for i in range(len(path)):
                children = Finder.find_children(deptree, path[i])
                sb_path.append('|'.join([ch for ch in children]))
                if i == len(path) - 1:
                    final_path += [token_map[path[i]]]
                else:
                    if not (path[i], path[i + 1]) in edges:
                        rela = (rel_map[(path[i + 1], path[i])], 'r')
                    else:
                        rela = (rel_map[(path[i], path[i + 1])], 'l')
                    final_path += [token_map[path[i]]]
                    final_path.append(rela)
        except Exception as e:
            print("Error")
        return final_path, sb_path

    def get_graph_feature(self, sent, deptree):
        index_by_content = {}
        index_token_offset = {}
        num_node = 0

        for token in sent.tokens:
            # print(token)
            index_by_content[token.content] = num_node
            index_token_offset[token.sent_offset[0]] = num_node
            # print("YY", token.content)
            num_node += 1

        adj = np.eye(num_node)

        for rel, pa, ch in deptree:
            u = index_token_offset[pa.sent_offset[0]]
            v = index_token_offset[ch.sent_offset[0]]
            adj[u][v] = 1.0
            adj[v][u] = 1.0

        adj2 = np.eye(num_node)

        for i in range(num_node):
            for j in range(num_node):
                for k in range(num_node):
                    if adj[i][j] == 1 and adj[j][k] == 1:
                        adj2[i][k] = 0.5
                        adj2[i][j] = 1
                        adj2[j][k] = 1

        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_matrix = np.diag(d_inv_sqrt)
        adj = d_inv_matrix.dot(adj)
        adj = adj.dot(d_inv_matrix)
        # for adj2
        rowsum = np.array(adj2.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_matrix = np.diag(d_inv_sqrt)
        adj2 = d_inv_matrix.dot(adj2)
        adj2 = adj2.dot(d_inv_matrix)

        x_feature = np.zeros((num_node, constants.INPUT_W2V_DIM))
        # if not is_random_embedding:
        embeddings = get_trimmed_w2v_vectors("relation_extraction/" + constants.TRIMMED_W2V)
        # else:
        #     embeddings = constants.random_embedding

        index_vocab = load_vocab("relation_extraction/" + constants.ALL_WORDS)
        i = 0
        for token in sent.tokens:
            # print("content : " ,token.content)
            index = index_vocab.get(token.content.lower(), index_vocab.get('$UNK$'))
            x_feature[i] = embeddings[index]
            i += 1

        X = self.ReLU(adj.dot(x_feature))
        for i in range(10):
            X = self.ReLU(adj.dot(X))

        return adj, adj2, X

    def ReLU(self, x):
        return abs(x) * (x > 0)


