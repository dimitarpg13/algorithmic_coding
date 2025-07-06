from typing import Iterable
from collections import Counter

class TraversalSolution:
    def __init__(self):
        self._reset_state()

    def _reset_state(self, s=None, k=None) -> None:
        self.k = k
        if s:
            self.n = len(s)
        else:
            self.n = None
        self.k_str_to_freq = dict() # strings with len k and their frequencies processed so far
        self.l_str_to_freq = dict() # strings with len larger than k and their frquencies processed so far

    @staticmethod
    def _char_frequencies(s: str) -> dict[str, int]:
        """
        type s: str
        rtype: dict[str, int]; str: char, int: count
        """
        return dict(Counter(s))
    
    @staticmethod
    def _is_subset(s: str, coll: Iterable[str]) -> bool:
        for coll_str in coll:
            if s in coll_str:
                return True
        return False 

    def _max_difference_subs(self, s: str, i: int) -> int:
        """
        type s: str, input string
        type i: int, start of the substring of s
        rtype: int, max difference between odd frequency and non-zero even frequency
           if the even frequency is 0 then return -len(s)-1  
        """
        subs = s[i:self.k+i]

        if subs not in self.k_str_to_freq:

            freq = self._char_frequencies(subs) 
            odds = set()
            evens = set()
            for val in freq.values():
                if val % 2 == 0:
                    evens.add(val)
                else:
                    odds.add(val)

            if len(odds) > 0:
                odd_max = max(odds)
            else:
                odd_max = 0
            
            even_min = 0
            if len(evens) > 0:
                even_min = min(evens)
                max_diff = odd_max - even_min
            else:
                max_diff = - self.n - 1
            
            self.k_str_to_freq[subs] = tuple([freq.copy(), odds.copy(), evens.copy(), odd_max, even_min, max_diff])
        else:
            freq, odds, evens, odd_max, even_min, max_diff = self.k_str_to_freq[subs]
            freq = freq.copy()
            odds = odds.copy()
            evens = evens.copy()

        chars_so_far = str(subs)
        for j in range(self.k+i, self.n):
            c = s[j]
            chars_so_far += c
            
            if not self._is_subset(chars_so_far, self.l_str_to_freq.keys()):

                if c in freq:
                    freq[c] += 1
                    if (freq[c] - 1) % 2 == 0:
                        odds.add(freq[c])
                        odd_max = max(odd_max, freq[c])
                
                        if freq[c] - 1 not in freq.values():
                            # recalculate the new min even freq
                            evens.remove(freq[c]-1)
                            if len(evens) > 0:
                                even_min = min(evens)
                            else:
                                even_min = 0

                    else:
                        evens.add(freq[c])
                        if even_min > 0:
                            even_min = min(even_min, freq[c])
                        else:
                            even_min = freq[c]

                        if freq[c] - 1 not in freq.values():
                            # recalculate the new max odd freq
                            odds.remove(freq[c]-1)
                            if len(odds) > 0:
                                odd_max = max(odds)
                            else:
                                odd_max = 0
                else:
                    freq[c] = 1
                    odds.add(1)
                    odd_max = max(odd_max, freq[c])
                
                if even_min > 0:
                    max_diff = max(max_diff, odd_max - even_min)
                
                self.l_str_to_freq[chars_so_far] = tuple([freq.copy(), odds.copy(), evens.copy(), odd_max, even_min, max_diff])
            else:
                freq, odds, evens, odd_max, even_min, max_diff = self.l_str_to_freq[chars_so_far]
                freq = freq.copy()
                odds = odds.copy()
                evens = evens.copy()

        return max_diff

    def max_difference(self, s: str, k: int) -> int:
        """
        type s: str
        """
        self._reset_state(s, k)

        if self.k > self.n:
            raise ValueError(f"invalid value for k: {self.k}")

        max_diff = - self.n - 1
        for i in range(0,self.n-self.k+1):
            max_diff = max(max_diff,self._max_difference_subs(s, i))

        # if max_diff == - self.n - 1:
        #    raise ValueError(f"No even frequency found with this input!") 
        return max_diff


if __name__ == '__main__':
    solution = TraversalSolution()

    s = "0001"
    k = 1
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))


    s = "1010313303"
    k = 1
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))

    s = "aabbccdde"
    k = 3
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
    
    s = "abcde"
    k = 2
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
    
    s = "aabbcc"
    k = 4
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
    
    s = "aaaa"
    k = 2
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
    
    s = "abcdabcd"
    k = 5
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
    
    s = "aabbccddeeffgghh"
    k = 6
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
    
    s = "xyzxyzxyz"
    k = 3
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
    
    s = "mnopmnopmnop"
    k = 4
    print("Max difference between even and odd counts in any substring of length least k:", solution.max_difference(s, k))
    
    s = "aabbccddeeffgghhiijj"
    k = 8
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
    
    s = "abcdefghij"
    k = 5
    print("Max difference between even and odd counts in any substring of length at least k:", solution.max_difference(s, k))
