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

    def _getMaxOddFrequency(freq1: dict, freq2: dict) -> int:
        """
        It merges the two frequency dicts and returns the new 
        """
        result = dict()
        n1 = len(freq1)
        n2 = len(freq2)
        if n1 < n2:



    def _maxDifferenceSubs(self, s: str, i: int) -> int:
        """
        type s: str, input string
        type i: int, start of the substring of s
        rtype: int, max difference between  
        """
        subs = s[i:self.k+i]
        freq = self._charFrequencies(subs) 
        odds = set()
        evens = set()
        even_min = 0
        for val in freq.values():
            if val % 2 == 0:
                evens.add(val)
            else:
                odds.add(val)
        if len(odds) > 0:
            odd_max = max(odds)
        else:
            odd_max = 0
        if len(evens) > 0:
            even_min = min(evens)
        else:
            # find if there is even count from position k+i+1 to the end of the string
            max_diff = None
            new_freq = dict()
            for j in range(self.k+i+2,self.n+1):
                subs2 = s[self.k+i:j]

                if subs[0] == subs[1] and subs[0] not in freq:
                    evens_min = min(evens)
                    max_diff = odds_max - evens_min
                    break
                else:
                    if not new_freq:
                        new_freq[subs[0]] = 1
                        new_freq[subs[1]] = 1
                    else:
                        if subs[-1] in new_freq:
                            new_freq[subs[-1]] += 1
                            evens_min = 2
                    
                            odds_max_new = self._getMaxOddFrequency(freq, new_freq)
                            if odds_max_new == odd_max:
                                max_diff = odds_max - evens_min
                                break
                            else:
                                if max_diff is None:
                                    max_diff = odds_max_new - evens_min
                                else:
                                    max_diff = max(max_diff, odds_max_new - evens_min)
                        else:
                            new_freq[subs[-1]] = 1
                
            if even_min == 0:
                raise ValueError(f"invalid input string! The even frequency count is zero: {s}")
        return max_diff

    def maxDifference(self, s: str, k: int) -> int:
        """
        type s: str
        """
        self.k = k
        self.n = len(s)

        max_diff = float('-inf')
        for i in range(0,self.n-self.k+1):
            max_diff = max(max_diff,self._maxDifferenceSubs(s, i))
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
