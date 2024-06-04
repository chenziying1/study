class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        if k >= len(num):
            return "0"
        if k == len(num) - 1:
            return str(min(map(int,list(num))))
        ans = pow(10,7)
        for i in range(len(num)-k):
            temp = num[i:i+k]
            ans = min(ans,int(temp))
        return str(ans)

num = "5337"
k = 2
print(Solution().removeKdigits(num,k))