import numpy as np
import time
import multiprocessing
from functools import partial

def worker(X, sfunc, X0):
    return sfunc(X0, X)

class Cluster:
    """
    Clustering by fast search and find of density peaks with improvements
    """
    def __init__(self, similarity_function, sc=0.5, n_core=1, verbose=False):
        self.verbose = verbose
        self.sc = sc
        
        self.n_core = n_core
        self.similarity_function = similarity_function

    def fit(self, Xs, is_sort=True):
        n_sample = len(Xs)

        if self.verbose:
            print("Start to cluster data by my own method with the size as [{0}] with sc = {1}.".format(n_sample, self.sc))

        if self.n_core > 1:
            return self.__fit_parallel(Xs)
        
        # if self.verbose:
        #     starttime = time.time()
        #     print("Calculating similarities with the data size as [{0}] .... ".format(n_sample),end="")

        # if ds is None:
        #     nXs = len(Xs)
        #     ds = np.zeros((nXs, nXs))
        #     for i in range(nXs):
        #         for j in range(i, nXs):
        #             ds[i,j] = self.distance_function(Xs[i], Xs[j])
        #             ds[j,i] = ds[i,j]
        # self.ds = ds

        # if self.verbose:
        #     endtime = time.time()
        #     print("Done. Dc = [{0}]. Spend time as [{1} seconds]".format(self.dc, np.around(endtime-starttime,3)))

        
        if self.verbose:
            starttime = time.time()
            print("Clustering ... ")

        center_indices = list()
        number_group = list()
        groups = list()

        # Add the first element to the group
        center_indices.append(0)
        number_group.append(1)
        groups.append([0])

        # cluster
        for i in range(1, n_sample):
            flag = True
            for n in range(len(center_indices)):
                index = center_indices[n]
                if self.similarity_function(Xs[i], Xs[index]) > self.sc:
                    # topologically same
                    number_group[n] += 1
                    groups[n].append(i)
                    flag = False
                    break
                    # topologically different

            if flag:
                if self.verbose:
                    print("A sample with new topological phase detected! [{0}]".format(i))
                center_indices.append(i)
                number_group.append(1)
                groups.append([i])
                    
        
        if self.verbose:
            endtime = time.time()
            print("Done. Spend time as [{0} seconds]".format(
                np.around(endtime-starttime,3)
            ))

        if is_sort:
            indices = np.argsort(-np.array(number_group))
            center_indices_sort = [center_indices[i] for i in indices]
            number_group_sort = [number_group[i] for i in indices]
            groups_sort = [groups[i] for i in indices]

            center_indices = center_indices_sort
            number_group = number_group_sort
            groups = groups_sort
        
        return center_indices, number_group, groups

    def __fit_parallel(self, Xs, is_sort=True):
        n_sample = len(Xs)

        if self.verbose:
            print("Start to cluster data by my own method parallel with the size as [{0}], sc = [{1}], the number of cores: [{2}].".format(n_sample, self.sc, self.n_core))

        
        if self.verbose:
            starttime = time.time()
            print("Clustering ... ")

        center_indices = list()
        number_group = list()
        groups = list()

        # Add the first element to the group
        center_indices.append(0)
        number_group.append(1)
        groups.append([0])

        # cluster
        for i in range(1, n_sample):
            f = partial(worker, sfunc=self.similarity_function, X0=Xs[i])
            centers = [Xs[n] for n in center_indices]

            pool = multiprocessing.Pool(self.n_core)
            similarities = pool.map(f, centers)
            pool.close()
            pool.join()

            similarities = np.array(similarities)
            flags = similarities > self.sc
            if np.any(flags):
                for n in range(len(center_indices)):
                    if flags[n]:
                        number_group[n] += 1
                        groups[n].append(i)
                        break

            else:
                if self.verbose:
                    print("A sample with new topological phase detected! [{0}]".format(i))

                center_indices.append(i)
                number_group.append(1)
                groups.append([i])
                    
        
        if self.verbose:
            endtime = time.time()
            print("Done. Spend time as [{0} seconds]".format(
                np.around(endtime-starttime,3)
            ))

        if is_sort:
            indices = np.argsort(-np.array(number_group))
            center_indices = [center_indices[i] for i in indices]
            number_group = [number_group[i] for i in indices]
            groups = [groups[i] for i in indices]
        
        return center_indices, number_group, groups

# def find_cluster_centers(rhos, deltas, Qs, dist_func, threshold=0.5, dc=0.1):
#     # indexs = [i for i in range(len(deltas)) if deltas[i]>threshold]
#     # center_indexs = list()
#     # for i in range(len(indexs)):
#     #     if len(center_indexs) == 0:
#     #         center_indexs.append(indexs[i])
#     #     else:
#     #         flag = 1
#     #         for center_index in center_indexs:
#     #             if np.abs((rhos[indexs[i]]-rhos[center_index])) < rho_width:
#     #                 flag = 0
#     #                 break
#     #         if flag:
#     #             center_indexs.append(indexs[i])

#     # return center_indexs
#     rhos, deltas = np.array(rhos), np.array(deltas)

#     center_indexs = list()
#     index_high_delta = [i for i in range(len(deltas)) if deltas[i]>threshold]
#     #print(index_high_delta)
#     indices = np.argsort(-rhos[index_high_delta])

#     # print(len(index_high_delta))
#     # print([deltas[i] for i in index_high_delta])

#     for i in indices:
#         index = index_high_delta[i]
#         flag = True
#         for c in center_indexs:
#             if dist_func(Qs[c], Qs[index]) < dc: 
#                 flag = False
#         if flag:
#             center_indexs.append(index)

#     return center_indexs