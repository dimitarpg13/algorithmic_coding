from collections import Counter

class TraversalSolution:
    def __init__(self):
        self.k = None
        self.n = None

    @staticmethod
    def _charFrequencies(s: str) -> dict[str, int]:
        """
        type s: str
        rtype: dict[str, int]; str: char, int: count
        """
        return dict(Counter(s))
        
    def _maxDifferenceSubs(self, s: str, i: int) -> int:
        """
        type s: str, input string
        type i: int, start of the substring of s
        rtype: int, max difference between odd frequency and non-zero even frequency
           if the even frequency is 0 then return -len(s)-1  
        """
        subs = s[i:self.k+i]
        freq = self._charFrequencies(subs) 
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
     
        for j in range(self.k+i, self.n):
            c = s[j]
            if c in freq:
                freq[c] += 1
                if (freq[c] - 1) % 2 == 0:
                    odds.add(freq[c])
                    odd_max = max(odd_max, freq[c])
            
                    if freq[c] - 1 not in freq.values():
                        # recalculate the new min even freq
                        if (freq[c]-1) in evens:
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
                        if (freq[c]-1) in odds:
                            odds.remove(freq[c]-1)
                        if len(odds) > 0:
                            odd_max = max(odds)
                        else:
                            odd_max = 0
            else:
                freq[c] = 1
                odd_max = max(odd_max, freq[c])
            
            if even_min > 0:
                max_diff = max(max_diff, odd_max - even_min)

        return max_diff

    def maxDifference(self, s: str, k: int) -> int:
        """
        type s: str
        """
        self.k = k
        self.n = len(s)
        if self.k > self.n:
            raise ValueError(f"invalid value for k: {self.k}")
        max_diff = - self.n - 1
        for i in range(0,self.n-self.k+1):
            max_diff = max(max_diff,self._maxDifferenceSubs(s, i))
        if max_diff == - self.n - 1:
            raise ValueError(f"No even frequency found with this input!") 
        return max_diff
            

if __name__ == '__main__':
    solution = TraversalSolution()
    s = "aabbccdde"
    k = 3
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "abcde"
    k = 2
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "aabbcc"
    k = 4
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "aaaa"
    k = 2
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "abcdabcd"
    k = 5
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "aabbccddeeffgghh"
    k = 6
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "xyzxyzxyz"
    k = 3
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "mnopmnopmnop"
    k = 4
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "aabbccddeeffgghhiijj"
    k = 8
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k))
    
    s = "abcdefghij"
    k = 5
    print("Max difference between even and odd counts in any substring of length k:", solution.maxDifference(s, k)) 
