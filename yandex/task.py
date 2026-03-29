import sys

sys.setrecursionlimit(2000)


input_data = sys.stdin.read().split()


iterator = iter(input_data)
n = int(next(iterator))
k = int(next(iterator))

is_valid = [False] * (n + 1)
is_valid[0] = True  
is_valid[n] = True  

for _ in range(k):
    try:
        x = int(next(iterator))
        if 0 <= x <= n:
            is_valid[x] = True
    except StopIteration:
        break
        
INF = n + 2 

dp_fwd = [INF] * (n + 1)
dp_fwd[0] = 0

for i in range(1, n + 1):
    if not is_valid[i]:
        continue
    
    if i - 1 >= 0 and is_valid[i-1] and dp_fwd[i-1] != INF:
        if dp_fwd[i-1] + 1 < dp_fwd[i]:
            dp_fwd[i] = dp_fwd[i-1] + 1
        
    if i - 2 >= 0 and is_valid[i-2] and dp_fwd[i-2] != INF:
        if dp_fwd[i-2] + 1 < dp_fwd[i]:
            dp_fwd[i] = dp_fwd[i-2] + 1
        
if dp_fwd[n] == INF:
    print("-1")
    sys.exit(0)
    
dp_bwd = [INF] * (n + 1)
dp_bwd[n] = 0

for i in range(n - 1, -1, -1):
    if not is_valid[i]:
        continue
        
    # Попытка прыжка в i+1
    if i + 1 <= n and is_valid[i+1] and dp_bwd[i+1] != INF:
        if dp_bwd[i+1] + 1 < dp_bwd[i]:
            dp_bwd[i] = dp_bwd[i+1] + 1
        
    if i + 2 <= n and is_valid[i+2] and dp_bwd[i+2] != INF:
        if dp_bwd[i+2] + 1 < dp_bwd[i]:
            dp_bwd[i] = dp_bwd[i+2] + 1
            
min_jumps = dp_fwd[n]
print(min_jumps)

curr = 0
path = []

while curr < n:
    found = False
    
    nxt = curr + 1
    if nxt <= n and is_valid[nxt]:
        if dp_fwd[nxt] == dp_fwd[curr] + 1 and dp_fwd[nxt] + dp_bwd[nxt] == min_jumps:
            path.append('1')
            curr = nxt
            found = True
    
    if not found:
        nxt = curr + 2
        if nxt <= n and is_valid[nxt]:
            if dp_fwd[nxt] == dp_fwd[curr] + 1 and dp_fwd[nxt] + dp_bwd[nxt] == min_jumps:
                path.append('2')
                curr = nxt
                found = True
    
    if not found:
        break
        
print("".join(path))