class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        for i in s:
            if stack and i in stack and i > stack[-1]:
                stack.pop(stack.index(i))
                stack.append(i)
            elif i not in stack:
                stack.append(i)
        return "".join(stack)

s = "cbacdcbc"
print(Solution().removeDuplicateLetters(s))