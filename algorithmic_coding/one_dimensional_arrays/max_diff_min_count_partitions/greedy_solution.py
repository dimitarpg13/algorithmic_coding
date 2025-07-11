from typing import List

class GreedySolution:
    MAX_VAL = 0
    MIN_VAL = 1
    MAX_IDX = 2
    MIN_IDX = 3

    def __init__(self):
        self.nums = None
        self.k = None 
        self.seqs = None
        self.seq_info = None
        # seqs are stored in a list of tuples in which the first
        # element encodes the seq min, the second element - the seq
        # max, and the third tuple element stores the sequence as a
        # list  

    def new_seq_info(self):
        return [float('-inf'), float('inf'), -1, -1]

    # Definition: max range of a sequence
    # We say that the max range R(S) of a sequence S = {s_1, s_2, ..., s_n} is K if
    # max(S) - min(S) = K.

    # Definition: Total range of a partition R_tot
    # The total range R_tot(P) of a partition P = {S_1, S_2, ..., S_m}
    # is given as R_tot(P) = sum_{j=1}^{m} R(S_j) 

    # Definition: "Less Than" relation for sequences:
    # for any pair of sequences S_1 and S_2 we say that S_1 < S_2 if every value 
    # in S_1 is smaller than every value in S_2. 

    # Definition: optimal k-partition
    # A partition of the original sequence S into subsequences S_1, S_2, ..., S_n
    # such that for each subsequence its max range is at most k and no partition
    # exist resulting in a smaller number of sequences having the specified range constraint.

    # Conjecture 1: We can always find an optimal k-partition in which for any two subsequences
    # S_1 and S_2  every element of one of the sequence is larger than every element of the
    # other.

    # Conjecture 2: We can always find an optimal k-partition in which the total range 
    # R_tot_opt is smaller than R_tot of any other k-partition

    # Algorithm: follows from Conjecture 1 and Conjecture 2
    # greedy strategy:
    # every new value is moved to the sequence to which mean it is closest to.
    # if the new value does not satisify the range constraint of all existing sequences
    # then we create a new sequence for it.  
    # Adjustment Procedure: Before creating a new sequence find out if we can rebalance the 
    # exisiting sequences such that the new value will be absorbed in one of the adjusted 
    # sequences.
    #  i) In each existing sequence identify a value on the oposite side of the range to 
    # that of the new element and try to rellocate it in another sequence without violating
    # the range constraint. 
    #  ii) Repeat until an adjusted sequence can absorb the new element or
    # until it becomes clear that absorbtion is not possible.   
    #
    # Conjecture 3: with the condition of Conjecture 2 enforced by the Algorithm we do not
    # need an Adjustment Procedure.
    # Proof: Suppose we have created p sequences so far - S1 < S2 < ... < Sp. 
    # We have a new value new_val which lies in between S_j range and S_{j+1} range, 1 <= j <= p  
    # The question is can this new value be absorbed in either S_j or S_{j+1} after adjustment?
    # If new_val is to be absorbed either into S_j or S_{j+1} then one of those needs to be adjusted.
    # Let us assume that S_j can be adjusted so that new_val can be absorbed. Obviously, in order this can
    # happen then we need to be able to remove the smallest value from S_j and add it to S_{j-1} without
    # violating the k-range constraint. But this cannot happen as if this was possible this value would 
    # have been already part of the sequence S_{j-1}. Using similar argument we show that S_{j+1} cannot
    # absorb new_val via adjustment. Hence the adjustment procedure is irrelevant with iterative construction
    # of the sequences S1 < S2 < ... < Sp.

    def find_seq_index(self, val: int, seqs: List[tuple]) -> int:
        l = len(seqs)
        start = 0
        end = l
        i = -1
        while start <= end:
            i = start + (end - start) // 2
            cur = seqs[i]
            if cur[self.MIN_VAL] <= val <= cur[self.MAX_VAL]:
                return i
            elif val < cur[self.MIN_VAL]:
                end = i - 1
            elif val > cur[self.MAX_VAL]:
                start = i + 1
        return i

    def partitionArray(self, nums: List[int], k: int) -> int:
        """
        nums: input array to be partitioned
        k: max difference between any pair of elements in  
        the same subsequence
        """
        self.nums = nums
        self.k = k
        self.seqs = list()
        self.seq_info = list()
        cur_seq = list()
        cur_seq_info = self.new_seq_info()
        prev_seq_info = None
        prev_seq = None
        for i, val in enumerate(nums):
            if i == 0:
                cur_seq.append(val)
                cur_seq_info[self.MIN_VAL] = val
                cur_seq_info[self.MAX_VAL] = val
                cur_seq_info[self.MIN_IDX] = 0
                cur_seq_info[self.MAX_IDX] = 0
                self.seqs.append(cur_seq)
                self.seq_info.append(cur_seq_info)
            else:
                if abs(val - cur_seq_info[self.MIN_VAL]) <= k and abs(val - cur_seq_info[self.MAX_VAL]) <= k:
                    if val < cur_seq_info[self.MIN_VAL]:
                        cur_seq_info[self.MIN_VAL] = val
                        cur_seq_info[self.MIN_IDX] = i 
                    elif val > cur_seq_info[self.MAX_VAL]:
                        cur_seq_info[self.MAX_VAL] = val
                        cur_seq_info[self.MAX_IDX] = i
                    cur_seq.append(val)
                elif val < cur_seq_info[self.MIN_VAL]:
                    idx = self.find_seq_index(val, self.seqs)
                    found_seq =self.seqs[idx]
                    found_seq_info = self.seq_info[idx]
                    if found_seq_info[self.MIN_VAL] <= val <= found_seq_info[self.MAX_VAL]:
                        # add the new value to the existing sequence
                        found_seq.append(val)

                    elif abs(found_seq_info[self.MIN_VAL]-val) <= k and abs(found_seq_info[self.MIN_VAL]-val) <= k:
                        # add the new value to the existing sequence and and update `cur_seq_info`
                        found_seq.append(val)
                        found_seq_info[self.MIN_VAL] = min(val, found_seq_info[self.MIN_VAL])
                        if found_seq_info[self.MIN_VAL] == val:
                            found_seq_info[self.MIN_IDX] = len(found_seq)-1
                        
                        found_seq_info[self.MAX_VAL] = max(val, found_seq_info[self.MAX_VAL])
                        if found_seq_info[self.MAX_VAL] == val:
                            found_seq_info[self.MAX_IDX] = len(found_seq)-1
                        
                    else: # we need to create a new sequence

                        prev_seq = cur_seq
                        prev_seq_info = cur_seq_info
                        cur_seq = list()
                        cur_seq_info = self.new_seq_info()
                        cur_seq_info[self.MIN_VAL] = val
                        cur_seq_info[self.MAX_VAL] = val
                        cur_seq_info[self.MIN_IDX] = 0
                        cur_seq_info[self.MAX_IDX] = 0

                        if val > found_seq_info[self.MAX_VAL]:
                            # found the place of the new sequence so create it and 
                            # insert it after index `idx`
                            self.seqs.insert(idx+1, cur_seq)
                            self.seq_info.insert(idx+1, cur_seq_info)


                        else:
                            # returns the first sequence in `self.seqs`` which is still larger 
                            # than the current value. So create a new sequence, insert it as the new
                            # first sequence in self.seqs and add the new value to it.
                            self.seqs.insert(idx, cur_seq)
                            self.seq_info.insert(idx, cur_seq)
                     
                elif val > cur_seq_info[self.MAX_VAL]:
                    # create a new sequence, append the new value to it, and append the new sequence
                    # to the end of `self.seqs`
                    prev_seq = cur_seq
                    prev_seq_info = cur_seq_info
                    cur_seq = list()
                    cur_seq_info = self.new_seq_info()
                    cur_seq.append(val)
                    cur_seq_info[self.MIN_VAL] = val
                    cur_seq_info[self.MAX_VAL] = val
                    cur_seq_info[self.MIN_IDX] = 0
                    cur_seq_info[self.MAX_IDX] = 0
                    self.seqs.append(cur_seq)
                    self.seq_info.append(cur_seq_info)

        return len(self.seqs)
  
if __name__ == '__main__':
    greedy_solution = GreedySolution()
    nums = [1, 2, 3, 4, 5, 6, 7, 8]
    k = 2
    print(f"Partitioning {nums} with k={k} results in {greedy_solution.partitionArray(nums, k)} sequences")
    
    nums = [1, 2, 3, 4, 5, 6, 7, 8]
    k = 3
    print(f"Partitioning {nums} with k={k} results in {greedy_solution.partitionArray(nums, k)} sequences")     

    nums = [3,1,3,4,2]
    k = 0
    print(f"Partitioning {nums} with k={k} results in {greedy_solution.partitionArray(nums, k)} sequences")
    