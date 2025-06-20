class Solution:
    def __init__(self):
        self.nums = None
        self.k = None 
        self.seqs = None
        # seqs are stored in a list of tuples in which the first
        # element encodes the seq min, tehe second element - the seq
        # max, and the third tuple element stores the sequence as a
        # list  

    # Conjecture: let s1 and s2 be any two subsequences resulting from
    # the partition of the original array `nums`. Then every element of 
    # one of the sequence is larger than every element of the other.
    


    def partitionArray(self, nums: List[int], k: int) -> int:
        """
        nums: input array to be partitioned
        k: max difference between any pair of elements in  
        the same subsequence
        """
        self.nums = nums
        self.k = k
        self.seqs = dict()
        cur_seq = list()   
        for i, val in enumerate(nums):
            if i == 0:
                cur_seq.append(val)
            else:
                pass
