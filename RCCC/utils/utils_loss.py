import torch
import torch.nn.functional as F

def cc_loss(outputs, partialY):
    # method 1：
    sm_outputs = F.softmax(outputs, dim=1)      #outputs are logits
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()     
    # method 2：
    # sm_outputs = F.log_softmax(outputs, dim=1)
    # final_outputs = sm_outputs * partialY
    # average_loss = - (final_outputs.sum(dim=1)).mean() 
    return average_loss
    
def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = - ((final_outputs).sum(dim=1)).mean()    #final_outputs=torch.Size([256, 10]) -> Q: why use negative? A:  
    return average_loss     #确实相当于CE loss的自实现版