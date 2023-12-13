import numpy as np
import torch
import time
# from thop import profile
# from thop import clever_format
def computeTime(model, device='cuda'):

    inputs = torch.randn(1,4, 3, 288, 288).cuda()
    if device == 'cuda':
        model = model.cuda()

    model.eval()
    # macs, params = profile(model, inputs=(inputs, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print("MACs is",macs)

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)
        if device == 'cuda':
            torch.cuda.synchronize()
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): %.4f, FPS:%d' % (np.mean(time_spent), 1*1//np.mean(time_spent) * 4))
    return 1*1//np.mean(time_spent)


if __name__=="__main__":

    torch.backends.cudnn.benchmark = True

    from model import MyModel
    model = MyModel()
    computeTime(model)
