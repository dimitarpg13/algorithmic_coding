class TraversalSolution:
    def __init__(self):
        self.nums = None
        self.k = None 
        self.seqs = None

    # Conjecture: let s1 and s2 be any two subsequences resulting from
    # the partition of the original array `nums`. Then every element of 
    # one of the sequence is larger than every element of the other.

    def partition_array(self, nums: List[int], k: int) -> int:
        """
        nums: input array to be partitioned
        k: max difference between any pair of elements in  
        the same subsequence
        """
        self.nums = nums
        self.k = k
        self.seqs = dict()   
        for i, val in enumerate(nums):
            pass
