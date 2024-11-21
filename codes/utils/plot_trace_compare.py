import numpy as np
import matplotlib.pyplot as plt
# salt = sio.loadmat('salt.mat')['data']
# MSE_unet_g30_dn = sio.loadmat('salt_sc100_MSE(unet-g30)_dn.mat')['data']
# vae_MSE_unet_g30_dn = sio.loadmat('salt_sc100_vae_MSE(unet-g30)_0_dn.mat')['data']
# MSE_unet_ng75_dn = sio.loadmat('salt_sc200_MSE(unet-ng75)_dn.mat')['data']


def plot_trace_compare(x,y,yf):
    trace_num_1= 60  # 60
    t=np.arange(0,128,1)
    plt.rcParams["font.family"] = "Times New Roman"
    fontsize=20
    linewidth=1.2
    fig, (ax0) = plt.subplots(nrows=1, figsize=(10,4))#ncols = 1
    ax0.plot(t, x[:, trace_num_1],'orange', label='clean trace of expert',linewidth=linewidth)
    # ax0.plot(t, yg[:, trace_num_1],'r', label='generated trace with Gaussion ',linewidth=linewidth)
    ax0.plot(t, y[:, trace_num_1], 'g'+'-',label='field trace',linewidth=linewidth)
    ax0.plot(t, yf[:, trace_num_1], 'b'+'-',label='generated field trace',linewidth=linewidth)

    ax0.set_xlabel("Time(ms)", fontsize=fontsize)
    ax0.set_ylabel("Amplitude", fontsize=fontsize)
    ax0.set_title('trace compare', fontsize=fontsize)# trace='+str(trace_num_1)
    ax0.legend(loc='lower right',fontsize=14)
    ax0.set_xticks(np.arange(0,128,32))
    ax0.set_xticklabels(np.arange(0,128,32), fontdict={'size':20})
    ax0.set_yticks(np.arange(-1,1.01,1))
    ax0.set_yticklabels(np.arange(-1,1.01,1), fontdict={'size':20})

    # axins01 = ax0.inset_axes((0.05, 0.1, 0.7, 0.25))
    # # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
    # axins01.plot(t,x[:, trace_num_1],'orange', label='clean trace',alpha=0.7)
    # # axins01.plot(t,dn[:, trace_num_1],'r', label='noisy trace',alpha=0.7)
    # axins01.plot(t,y[:, trace_num_1],'g'+'-',label='$\widetilde{\\rm{\mathbb{G}}}$',alpha=0.7)
    # axins01.plot(t,yf[:, trace_num_1],'b'+'-',label='SRNGF($\widetilde{\\rm{\mathbb{G}}}$)',alpha=0.7)
    # axins01.set_xticks(np.arange(0,128,16))
    # axins01.set_xticklabels(np.arange(0,128,16), fontdict={'size':10})
    # axins01.grid()
    # # 局部显示并且进行连线
    # from utils.zone_and_linked import zone_and_linked
    # zone_and_linked(ax0, axins01, 0, 127, t , [x[:, trace_num_1],
    #                                            y[:, trace_num_1],yf[:, trace_num_1]], 'bottom')
    fig.tight_layout()
    fig.set_dpi(400)
    plt.show()
