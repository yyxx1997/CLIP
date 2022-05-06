import torch
from torch import tensor
 
a = tensor([[1, 2, 3, 4],
        [1, 2, 3, 4]]).float()  #norm仅支持floatTensor,a是一个2*4的Tensor
a0 = torch.norm(a,p=2,dim=0,keepdim=True)    #按0维度求2范数
a1 = torch.norm(a,p=2,dim=1,keepdim=True)    #按1维度求2范数
print(a0)
print(a1)
print(a/a1)
sm_a = torch.softmax(a,dim=1)
sm_a0 = torch.softmax(a/a1,dim=1)

print(sm_a)
print(sm_a0)