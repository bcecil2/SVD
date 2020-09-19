import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np


def bwApprox(A,tol):
    u,s,v = np.linalg.svd(A,full_matrices=False)

    uprime = u[...,:tol]
    sprime = np.diag(s[:tol])
    vprime = v[:tol]


    size = np.linalg.matrix_rank(sprime)
    final = np.matmul(np.matmul(uprime,sprime),vprime)
    
    return final,size

def colorApprox(A,tol):
    m,n,d = A.shape
    rgb = np.dsplit(A,3)
    
    rSVD, rSize = bwApprox(np.reshape(rgb[0],(m,n)),tol)
    gSVD, gSize = bwApprox(np.reshape(rgb[1],(m,n)),tol)
    bSVD, bSize = bwApprox(np.reshape(rgb[2],(m,n)),tol)
    return np.dstack((rSVD, gSVD, bSVD)),(rSize,gSize,bSize) 


img = skimage.io.imread("./Lenna.png")

print(img.shape)
shape = img.shape
if len(shape) == 2:
    d = 2
else:
    d = 3
m = shape[0]
n = shape[1]

compImg = None

fig, axes = plt.subplots(2, 2, figsize=(8, 4))
ax = axes.ravel()
tol = m//15
for i in range(len(ax)):
    if i == 0:
        ax[i].set_title("Original")
        ax[i].set_xlabel("Original Pixel Count " + str(img.size))
        if d == 2:
            ax[i].imshow(img ,cmap="gray")
        elif d == 3:
            ax[i].imshow(img)

        ax[i].set_xticks([])
        ax[i].set_yticks([])

    else:
        if d == 2:
            compImg, size = bwApprox(img,tol)
        elif d == 3:
            compImg, size = colorApprox(img,tol) 
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if d == 2:
            ax[i].imshow(compImg / 255 , cmap="gray")
        elif d == 3:
            ax[i].imshow(compImg / 255)

        
        ax[i].set_title("Rank " + str(tol) + " approximation")
        tol += tol

fig.tight_layout()
plt.show()
