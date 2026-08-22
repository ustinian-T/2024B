# -*- coding: utf-8 -*-
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
with open('2024B.tex','r',encoding='utf-8') as f:
    lines = f.readlines()

envs = ['enumerate','itemize','tabular','tabularx','longtable','figure','table','equation','aligned','cases','abstract','appendices','document','item']
stack = []
for i,l in enumerate(lines,1):
    # skip commented lines
    s = l.split('%')[0]
    for m in re.finditer(r'\\begin\{(\w+)\}', s):
        env = m.group(1)
        stack.append((env,i))
    for m in re.finditer(r'\\end\{(\w+)\}', s):
        env = m.group(1)
        # pop matching
        if stack and stack[-1][0]==env:
            stack.pop()
        else:
            # mismatch
            print(f'行{i}: \\end{{{env}}} 但栈顶是 {stack[-1] if stack else "空"}')
            # try to find it
            for k in range(len(stack)-1,-1,-1):
                if stack[k][0]==env:
                    del stack[k]
                    print(f'  从栈中移除位置{k}的 {env}')
                    break
print('===未闭合的(栈中剩余)===')
for env,ln in stack:
    if env in ['enumerate','itemize','tabular','tabularx','longtable','figure','table','equation','aligned','cases','abstract','document','appendices']:
        print(f'  {env} 开始于行{ln} 未闭合')
