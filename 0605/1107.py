from typing import List
import heapq

class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        if not nums1 or not nums2 or k <= 0:
            return []
        min_heap = []
        for i in range(min(len(nums1), k)):
            heapq.heappush(min_heap, (nums1[i] + nums2[0], i, 0))
        result = []
        while k > 0 and min_heap:
            _, i, j = heapq.heappop(min_heap)
            result.append([nums1[i], nums2[j]])
            if j + 1 < len(nums2):
                heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))
            k -= 1
        
        return result
nums1 = [1,7,11]
nums2 = [2,4,6]
k = 3
print(Solution().kSmallestPairs(nums1=nums1, nums2=nums2, k=k))