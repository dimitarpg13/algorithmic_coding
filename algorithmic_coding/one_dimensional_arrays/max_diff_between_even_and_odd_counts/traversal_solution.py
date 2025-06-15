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
            for j in range(self.k+i+2,self.n+1):
                subs2 = s[k+i:j]
                if subs[0] == subs[1]:
                    evens_min = min(evens)
                    break
            
            if even_min == 0:
                raise ValueError(f"invalid input string {s}")
        
        return odd_max - even_min

    def maxDifference(self, s: str, k: int) -> int:
        """
        type s: str
        """
        self.k = k
        self.n = len(s)

        max_diffs = list()
        for i in range(0,self.n-self.k+1):
            max_diffs.append(self._maxDifferenceSubs(s, i))
        return max(max_diffs)
            

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
