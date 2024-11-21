import numpy as np
def show_GT_NY_Dn_Rn_GTn_Resi(x,y,x_):
    import matplotlib.pyplot as plt
    clip = abs(y).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    plt.figure(figsize=(30, 10))  # 16,3
    plt.subplot(161)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)  # cmap='gray'
    plt.title('clean data')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(162)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('noisy data')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(163)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('denoised data')
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('./noise/vdn_dn_uns_25.mat'), {'data': x_[:, :, np.newaxis]})

    plt.subplot(164)
    noise = y - x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('removed noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(165)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('groundtruth noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(166)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_ - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('residual')
    # plt.colorbar(shrink=0.5)
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})


def show_GT_NY_Dn_Rn(x,y,x_,dpi,figsize):
    import matplotlib.pyplot as plt
    clip = abs(y).max()  # 显示范围，负值越大越明显
    clip=1
    vmin, vmax = -clip, clip
    plt.figure(dpi=dpi,figsize=figsize)  # 16,3
    plt.subplot(141)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)  # cmap='gray'
    plt.title('clean data')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(142)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('noisy data')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(143)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('denoised data')
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('./noise/vdn_dn_uns_25.mat'), {'data': x_[:, :, np.newaxis]})

    plt.subplot(144)
    noise = y - x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('removed noise')
    # plt.colorbar(shrink=0.5)

    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})

def show_NY_Dn_Rn(y,x_):
    import matplotlib.pyplot as plt
    clip = abs(y).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    plt.figure(figsize=(20, 10))  # 16,3

    plt.subplot(131)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('noisy data')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(132)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('denoised data')
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('./noise/vdn_dn_uns_25.mat'), {'data': x_[:, :, np.newaxis]})

    plt.subplot(133)
    noise = y - x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('removed noise')
    # plt.colorbar(shrink=0.5)

    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})

def show_GT_GN_FN_RN(GT,GN,FN,RN):
    import matplotlib.pyplot as plt
    clip = abs(RN).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    # plt.figure(figsize=(20, 10))  # 16,3  20, 10

    plt.figure(dpi=300, figsize=(20, 6)) #(12, 6)
    plt.subplot(1, 4, 1)
    plt.imshow(GT, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Clean Image')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    # GN = GT+ np.random.normal(0, 1, GT.shape) * 0.1
    plt.imshow(GN, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Clean Image with Gaussian')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(FN, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Generated Fake Noisy Image')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(RN, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Real Noisy Image')
    plt.axis('off')
    plt.show()

def show_GC_GN_FN_RN(GC,GN,FN,RN):
    import matplotlib.pyplot as plt
    clip = abs(RN).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    # plt.figure(figsize=(20, 10))  # 16,3  20, 10

    plt.figure(dpi=300, figsize=(20, 6)) #(12, 6)
    plt.subplot(1, 4, 1)
    plt.imshow(GC, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Clean Image')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    # GN = GC+ np.random.normal(0, 1, GC.shape) * 0.1
    plt.imshow(GN, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Clean Image with Gaussian')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(FN, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Generated Fake Noisy Image')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(RN, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Real Noisy Image')
    plt.axis('off')
    plt.show()

def show_FN_RN(FN,RN):
    import matplotlib.pyplot as plt
    clip = abs(RN).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    # plt.figure(figsize=(20, 10))  # 16,3  20, 10

    plt.figure(dpi=300, figsize=(10, 6)) #(12, 6)
    plt.subplot(1, 2, 1)
    plt.imshow(FN, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Generated Fake Noisy Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(RN, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.title('Real Noisy Image')
    plt.axis('off')
    plt.show()