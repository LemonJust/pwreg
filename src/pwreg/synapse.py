import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from .core import Points

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})


# from synspy> analyze > pair :
def nearest_pairs(v1, kdt1, v2, radius, out1, out2):
    """Find nearest k-dimensional point pairs between v1 and v2 and return via output arrays.

       Inputs:
         v1: array with first pointcloud with shape (m, k)
         kdt1: must be cKDTree(v1) for correct function
         v2: array with second pointcloud with shape (m, k)
         radius: maximum euclidean distance between points in a pair
         out1: output adjacency matrix of shape (n,)
         out2: output adjacency matrix of shape (m,)

       Use greedy algorithm to assign nearest neighbors without
       duplication of any point in more than one pair.

       Outputs:
         out1: for each point in kdt1, gives index of paired point from v2 or -1
         out2: for each point in v2, gives index of paired point from v1 or -1

    """
    depth = min(max(out1.shape[0], out2.shape[0]), 100)
    out1[:] = -1
    out2[:] = -1
    # print(f"v2 shape : {v2.shape}")
    dx, pairs_nn = kdt1.query(v2, depth, distance_upper_bound=radius)

    # arrays need to be 2D :
    if pairs_nn.shape == (1,):
        pairs_nn = pairs_nn[:, None]
    if dx.shape == (1,):
        dx = dx[:, None]
    if out1.shape == (1,):
        out1 = out1[:, None]
    if out2.shape == (1,):
        out2 = out2[:, None]

    for d in range(depth):
        for idx2 in np.argsort(dx[:, d]):
            if dx[idx2, d] < radius:
                if out2[idx2] == -1 and out1[pairs_nn[idx2, d]] == -1:
                    out2[idx2] = pairs_nn[idx2, d]
                    out1[pairs_nn[idx2, d]] = idx2


def pair(centroids1, centroids2, radius_seq):
    ve1 = centroids1  # tp1
    ve2 = centroids2  # tp2

    kdt1 = cKDTree(ve1)
    v1_to_v2 = np.zeros((len(radius_seq), ve1.shape[0]), dtype=np.int32)
    v2_to_v1 = np.zeros((len(radius_seq), ve2.shape[0]), dtype=np.int32)
    # print(f"v1_to_v2: {v1_to_v2.shape}")
    # print(f"v2_to_v1: {v2_to_v1.shape}")

    for r_idx in range(len(radius_seq)):
        nearest_pairs(ve1, kdt1, ve2, radius_seq[r_idx], v1_to_v2[r_idx, :], v2_to_v1[r_idx, :])

    return v1_to_v2, v2_to_v1


# plotting functions
def pair_summary(v1_to_v2, title_intro):
    paired = np.zeros((len(radius_seq),), dtype=np.int32)
    n_total = reg[study_id]['tp1']['xyz'].shape[0]
    for r_idx in range(len(radius_seq)):
        paired[r_idx] = np.sum(v1_to_v2[r_idx, :] != -1)
    unpaired = n_total - paired

    paired = paired * 100 / n_total
    unpaired = unpaired * 100 / n_total

    print(f'radius :{radius_seq} \npaired %:\n{paired} \nunpaired %:\n{unpaired}')

    fig = plt.figure(dpi=160)
    plt.plot(radius_seq, paired)
    plt.vlines(4, 0, 100, colors='r')
    plt.grid()
    plt.title(f'{title_intro}\nNumber of paired synapses')
    plt.xlabel('Deforestation radius, um')
    plt.ylabel('Synapses paired, %')


class UnchangedSynapsePair:
    """
    Keeps all the extra info about the pairs.
    """

    def __init__(self, centroids1, centroids2, pairing, note=None):
        """
        centroids1, centroids2 : centroids in order xyz, shape (N,3) for tp1 and tp2
        pairing (list ) : pairing of what synapses from tp1 are paired with tp2, -1 if no pair.
                            Contains indices of synapses at tp2 in order of synapses at tp1.
        note (str) : tells what space the centroids are, order xyz or zyx etc ...
        """
        has_pair = pairing != -1
        self.num_pairs = np.sum(has_pair)
        self.idx = {'tp1': np.where(has_pair)[0],
                    'tp2': pairing[has_pair]}
        self.centroids = {'tp1': centroids1[self.idx['tp1']],
                          'tp2': centroids2[self.idx['tp2']]}

        self.diff = self.centroids['tp1'] - self.centroids['tp2']
        self.distance = np.linalg.norm(self.diff, axis=1)

        self.min_distance = np.min(self.distance)
        self.max_distance = np.max(self.distance)
        self.note = note

        self.correlation = {}


class SynapsePair:
    """
    Keeps all the info about unchanged, lost, gained.
    """

    def __init__(self, ptc1, ptc2, radius, match=None):
        """
        For now fixed : units='phs'

        centroids1, centroids2 : centroids in order zyx, shape (N,3) for tp1 and tp2
        pairing1 (list ) : pairing of what synapses from tp1 are paired with tp2, -1 if no pair.
                            Contains indices of synapses at tp2 in order of synapses at tp1.
        pairing2 (list ) : pairing of what synapses from tp2 are paired with tp1, -1 if no pair.
                            Contains indices of synapses at tp1 in order of synapses at tp2.
        note (str) : tells what space the centroids are, order xyz or zyx etc ...
        """

        self.ptc1 = ptc1
        self.ptc2 = ptc2

        self.radius = radius

        if match is None:
            match = self.prepare_match()
        self.match = match

        # print(f"numb numa : {ptc1.num_points}, {ptc2.num_points}")

        self.uncB, self.uncA, self.lost, self.gained = self.pair()

        # fraction lost and gained
        self.fr_lost = self.lost.num_points / self.ptc1.num_points
        self.fr_gained = self.gained.num_points / self.ptc1.num_points

    def prepare_match(self):
        """
        Prepares a dictionary with status -2 for all the synapses at tp1 and tp2
        """
        return {'per_idx1': np.ones((self.ptc1.num_points,)) * (-2),
                'per_idx2': np.ones((self.ptc2.num_points,)) * (-2)}

    def pair(self):
        """
        returns Points for unchanged lost and gained synapses
        """
        pairing1, pairing2 = pair(self.ptc1.zyx['phs'], self.ptc2.zyx['phs'], [self.radius])
        # we only use one radius anyway, but grab the first column ( alternatively you could squeeze ..)
        pairing1 = pairing1[0]
        pairing2 = pairing2[0]

        # TODO : do this after getting lost/ gained/ unchanged and use those idx directly
        for i_syn, syn in enumerate(self.ptc1.idx):
            if pairing1[i_syn] == -1:
                self.match['per_idx1'][syn] = -1
            else:
                self.match['per_idx1'][syn] = self.ptc2.idx[pairing1[i_syn]]

        for i_syn, syn in enumerate(self.ptc2.idx):
            if pairing2[i_syn] == -1:
                self.match['per_idx2'][syn] = -1
            else:
                self.match['per_idx2'][syn] = self.ptc1.idx[pairing2[i_syn]]

        has_pair = pairing1 != -1
        paired_tp1_idx = np.where(has_pair)[0]
        paired_tp2_idx = pairing1[has_pair]
        lost_idx = np.where(pairing1 == -1)[0]
        gained_idx = np.where(pairing2 == -1)[0]

        uncB = Points(self.ptc1.zyx['phs'][paired_tp1_idx], units='phs',
                      resolution=self.ptc1.resolution, idx=self.ptc1.idx[paired_tp1_idx])
        uncA = Points(self.ptc2.zyx['phs'][paired_tp2_idx], units='phs',
                      resolution=self.ptc2.resolution, idx=self.ptc2.idx[paired_tp2_idx])
        lost = Points(self.ptc1.zyx['phs'][lost_idx], units='phs',
                      resolution=self.ptc1.resolution, idx=self.ptc1.idx[lost_idx])
        gained = Points(self.ptc2.zyx['phs'][gained_idx], units='phs',
                        resolution=self.ptc2.resolution, idx=self.ptc2.idx[gained_idx])

        return uncB, uncA, lost, gained

    def summary(self):
        summary = f"Total before: {self.ptc1.num_points}\n" \
                  f"Total after: {self.ptc2.num_points}\n" \
                  f"Fraction lost: {self.fr_lost}\n" \
                  f"Fraction gained: {self.fr_gained}"
        print(summary)


class PwSynapsePair:

    def __init__(self, ptc1_list, ptc2_list, radius, numB, numA):
        self.ptc1 = ptc1_list
        self.ptc2 = ptc2_list
        self.radius = radius  # in um

        # total number of synapses before and after
        self.numB = numB
        self.numA = numA

        self.uncB, self.uncA, self.lost, self.gained, self.match = self.pair()

    def pair(self):
        uncB = []
        uncA = []
        lost = []
        gained = []
        match = {'per_idx1': [], 'per_idx2': []}

        for ptc1, ptc2 in tqdm(zip(self.ptc1, self.ptc2), total=len(self.ptc1)):
            match_block = {'per_idx1': [-2] * self.numB, 'per_idx2': [-2] * self.numA}
            if not (ptc1.num_points == 0 and ptc2.num_points == 0):
                if ptc1.num_points == 0 and (not ptc2.num_points == 0):
                    for idx in ptc2.idx:
                        match_block['per_idx2'][idx] = -1
                elif (not ptc1.num_points == 0) and ptc2.num_points == 0:
                    for idx in ptc1.idx:
                        match_block['per_idx1'][idx] = -1
                else:
                    sp = SynapsePair(ptc1, ptc2, self.radius, match=match_block)
                    uncB.append(sp.uncB)
                    uncA.append(sp.uncA)
                    lost.append(sp.lost)
                    gained.append(sp.gained)
                    match_block = sp.match
            # will be None if all are empty
            match['per_idx1'].append(match_block['per_idx1'])
            match['per_idx2'].append(match_block['per_idx2'])

        match['per_idx1'] = np.stack(match['per_idx1']).T
        match['per_idx2'] = np.stack(match['per_idx2']).T

        return uncB, uncA, lost, gained, match

    def mask_by_block(self, blcs):
        """
        Split ptc1 and ptc2_list into the corresponding blocks,
        leave only the match for the synapses inside the blocks of interest.
        """
        # get point clouds cropped to the blocks in fixed space
        ptc1_crop = self.ptc1
        ptc2_crop = [ptc2.fit_block(blc) for ptc2, blc in zip(self.ptc2, blcs)]

        # mask the matches
        pass
