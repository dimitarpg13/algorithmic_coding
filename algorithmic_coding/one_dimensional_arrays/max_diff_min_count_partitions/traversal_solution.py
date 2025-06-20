from typing import List

class Solution:
    def __init__(self):
        self.nums = None
        self.k = None 
        self.seqs = None
        # seqs are stored in a list of tuples in which the first
        # element encodes the seq min, the second element - the seq
        # max, and the third tuple element stores the sequence as a
        # list  

    # Definition: optimal partition
    # A partition of the original sequence s into subsequences s1, s2, ..., sn
    # such that for each subsequence the max range within it is at most k and no partition
    # exist resulting in a smaller number of sequences having this range constraint.

    # Conjecture: We can always find an optimal partition in which for any two subsequences
    # s1 and s2  every element of one of the sequence is larger than every element of the
    # other.
    
    def find_prev_seq(val: int, seqs: List[tuple]) -> tuple:
        for cur in reversed(seqs):
            if cur[0] <= val <= cur[1]:

    def partitionArray(self, nums: List[int], k: int) -> int:
        """
        nums: input array to be partitioned
        k: max difference between any pair of elements in  
        the same subsequence
        """
        self.nums = nums
        self.k = k
        self.seqs = list()
        cur_seq = list()
        cur_min = float('inf')
        cur_max = float('-inf')
        cur_min_idx = -1
        cur_max_idx = -1 
        prev_seq = None
        for i, val in enumerate(nums):
            if i == 0:
                cur_seq.append(val)
                cur_min = val
                cur_max = val
                cur_min_idx = 0
                cur_max_idx = 0
            else:
                 if abs(val - cur_min) <= k and abs(val - cur_max) <= k:
                     if val < cur_min:
                        cur_min = val
                        cur_min_idx = i 
                     elif val > cur_max:
                        cur_max = val
                        cur_max_idx = i
                     cur_seq.append(val)
                 elif val < cur_min:
                     
                 elif val > cur_max:



