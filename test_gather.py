import torch
a = [1,2,3,4,5]
b = [6,7,8,9,10]
c = [11,12,13,14,15]

src = torch.tensor([[a,b,c],[a,b,c]])
trg = torch.tensor([[[0],[1]],[[1],[2]]]) #第一个样本取t=0的第0维度，t=1的第1维度，第二个样本取t=0的第1维度, t=1的第二维度
print(src)
print(torch.gather(src,-1,trg))
trg = torch.tensor([[[1,1,1,1,1]],[[2,2,2,2,2]]]) #第一个sample取t=1，第二个sample取t=2, 每个t取所有维度，
print(torch.gather(src,1,trg))
indice = torch.tensor([[0,1],[1,2]])
print(src[torch.arange(src.size(0)),indice[None,:],:])
print(src[torch.arange(src.size(0))[:,None],indice[None,:],:].shape)

print("src",src.shape)
mask = torch.tensor([[0,0,1],[0,1,0]])
print(src * mask.unsqueeze(-1))

