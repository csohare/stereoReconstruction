import numpy as np
import matplotlib.pyplot as plt
import camutils as cu
import pickle
import visutils2 as visutils
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import trimesh
import cv2

def writeply(X,color,tri,filename):

    f = open(filename,"w")
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex %i\n' % X.shape[1])
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('element face %d\n' % tri.shape[0])
    f.write('property list uchar int vertex_indices\n')
    f.write('end_header\n')

    C = (255*color).astype('uint8')
    i = 0
    
    for i in range(X.shape[1]):
        f.write('%f %f %f %i %i %i\n' % (X[0,i],X[1,i],X[2,i],C[0,i],C[1,i],C[2,i]))
    
    for t in range(tri.shape[0]):
        f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))

    f.close()


def grab_color(file1, file2, coords0, coords1):
    p1 = plt.imread(file1)
    p2 = plt.imread(file2)

    c1 = p1[coords0[1], coords0[0]].T
    c2 = p2[coords1[1], coords1[0]].T
    
    return ((c1 + c2) * 1.5) / 2.0
    

def reconstruct(imprefixL,imprefixR,threshold,camL,camR):

    CLh,maskLh = decode(imprefixL,0,threshold)
    CLv,maskLv = decode(imprefixL,20,threshold)
    CRh,maskRh = decode(imprefixR,0,threshold)
    CRv,maskRv = decode(imprefixR,20,threshold)

    CL = CLh + 1024*CLv
    maskL = maskLh*maskLv
    CR = CRh + 1024*CRv
    maskR = maskRh*maskRv

    h = CR.shape[0]
    w = CR.shape[1]

    subR = np.nonzero(maskR.flatten())
    subL = np.nonzero(maskL.flatten())

    CRgood = CR.flatten()[subR]
    CLgood = CL.flatten()[subL]

    _,submatchR,submatchL = np.intersect1d(CRgood,CLgood,return_indices=True)

    matchR = subR[0][submatchR]
    matchL = subL[0][submatchL]

    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))

    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)

    pts3 = triangulate(pts2L,camL,pts2R,camR)

    return pts2L,pts2R,pts3


def decode(imprefix,start,threshold):

    nbits = 10
    
    imgs = list()
    imgs_inv = list()
    print('loading',end='')
    for i in range(start,start+2*nbits,2):
        fname0 = '%s%2.2d.png' % (imprefix,i)
        fname1 = '%s%2.2d.png' % (imprefix,i+1)
        print('(',i,i+1,')',end='')
        img = plt.imread(fname0)
        img_inv = plt.imread(fname1)
        if (img.dtype == np.uint8):
            img = img.astype(float) / 256
            img_inv = img_inv.astype(float) / 256
        if (len(img.shape)>2):
            img = np.mean(img,axis=2)
            img_inv = np.mean(img_inv,axis=2)
        imgs.append(img)
        imgs_inv.append(img_inv)
        
    (h,w) = imgs[0].shape
    print('\n')
    
    gcd = np.zeros((h,w,nbits))
    mask = np.ones((h,w))
    for i in range(nbits):
        gcd[:,:,i] = imgs[i]>imgs_inv[i]
        mask = mask * (np.abs(imgs[i]-imgs_inv[i])>threshold)
        
    bcd = np.zeros((h,w,nbits))
    bcd[:,:,0] = gcd[:,:,0]
    for i in range(1,nbits):
        bcd[:,:,i] = np.logical_xor(bcd[:,:,i-1],gcd[:,:,i])
        
    code = np.zeros((h,w))
    for i in range(nbits):
        code = code + np.power(2,(nbits-i-1))*bcd[:,:,i]
        
    return code,mask

#CAMERA CALIBRATION 
fid = open('calibration.pickle','rb')
calib = pickle.load(fid)

f = np.average(np.array([calib["fx"], calib["fy"]]));
t = np.zeros((3, 1))
r = np.identity(3)
c = np.array([[calib["cx"], calib["cy"]]]).T


camL = cu.Camera(f, c, t, r)
camR = cu.Camera(f, c, t, r) 

imgL = plt.imread('calib_jpg_u/frame_C0_01.jpg')
ret, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
pts2L = cornersL.squeeze().T

imgR = plt.imread('calib_jpg_u/frame_C1_01.jpg')
ret, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
pts2R = cornersR.squeeze().T

pts3 = np.zeros((3,6*8))
xx,yy = np.meshgrid(np.arange(8),np.arange(6))
pts3[0,:] = 2.8*yy.reshape(1,-1)
pts3[1,:] = 2.8*xx.reshape(1,-1)

paramsL_init = np.array([180,0,0,0,0,10])
paramsR_init = np.array([180,0,0,0,0,10])

camL = cu.calibratePose(pts3,pts2L,camL,paramsL_init)
camR = cu.calibratePose(pts3,pts2R,camR,paramsR_init)

pts3r = cu.triangulate(pts2L,camL,pts2R,camR)


plt.rcParams['figure.figsize']=[15,15]
pts2Lp = camL.project(pts3)
plt.imshow(imgL)
plt.plot(pts2Lp[0,:],pts2Lp[1,:],'bo')
plt.plot(pts2L[0,:],pts2L[1,:],'rx')
plt.show()

pts2Rp = camR.project(pts3)
plt.imshow(imgR)
plt.plot(pts2Rp[0,:],pts2Rp[1,:],'bo')
plt.plot(pts2R[0,:],pts2R[1,:],'rx')
plt.show()

print(camL)
print(camR)
print(np.sqrt(np.sum((camL.t-camR.t)*(camL.t-camR.t))))

print(np.mean(np.sqrt(np.sum((pts3r-pts3)*(pts3r-pts3),axis=0))))

print(np.mean(np.sum((pts3r-pts3)*(pts3r-pts3),axis=0)))
print("Average error:", np.average(pts3 - pts3r))
print(f'The average error of my recovered 3D locations of the grid corner points relative to their true coordinates is: {np.mean(np.sqrt(np.sum((pts3r-pts3)*(pts3r-pts3),axis=0)))} cm')
print(np.mean(np.absolute(pts3r - pts3)))

lookL = np.hstack((camL.t,camL.t+camL.R @ np.array([[0,0,2]]).T))
lookR = np.hstack((camR.t,camR.t+camR.R @ np.array([[0,0,2]]).T))


fig = plt.figure()
ax = fig.add_subplot(2,2,1,projection='3d')
ax.plot(pts3[0,:],pts3[1,:],pts3[2,:],'.')
ax.plot(pts3r[0,:],pts3r[1,:],pts3r[2,:],'rx')
ax.plot(camR.t[0],camR.t[1],camR.t[2],'ro')
ax.plot(camL.t[0],camL.t[1],camL.t[2],'bo')
ax.plot(lookL[0,:],lookL[1,:],lookL[2,:],'b')
ax.plot(lookR[0,:],lookR[1,:],lookR[2,:],'r')
visutils.set_axes_equal_3d(ax)
visutils.label_axes(ax)
plt.title('scene 3D view')

ax = fig.add_subplot(2,2,2)
ax.plot(pts3[0,:],pts3[2,:],'.')
ax.plot(pts3r[0,:],pts3r[2,:],'rx')
ax.plot(camR.t[0],camR.t[2],'ro')
ax.plot(camL.t[0],camL.t[2],'bo')
ax.plot(lookL[0,:],lookL[2,:],'b')
ax.plot(lookR[0,:],lookR[2,:],'r')
plt.title('XZ-view')
plt.grid()
plt.xlabel('x')
plt.ylabel('z')

ax = fig.add_subplot(2,2,3)
ax.plot(pts3[1,:],pts3[2,:],'.')
ax.plot(pts3r[1,:],pts3r[2,:],'rx')
ax.plot(camR.t[1],camR.t[2],'ro')
ax.plot(camL.t[1],camL.t[2],'bo')
ax.plot(lookL[1,:],lookL[2,:],'b')
ax.plot(lookR[1,:],lookR[2,:],'r')
plt.title('YZ-view')
plt.grid()
plt.xlabel('y')
plt.ylabel('z')

ax = fig.add_subplot(2,2,4)
ax.plot(pts3[0,:],pts3[1,:],'.')
ax.plot(pts3r[0,:],pts3r[1,:],'rx')
ax.plot(camR.t[0],camR.t[1],'ro')
ax.plot(camL.t[0],camL.t[1],'bo')
ax.plot(lookL[0,:],lookL[1,:],'b')
ax.plot(lookR[0,:],lookR[1,:],'r')
plt.title('XY-view')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')

def points3d (filepath1, filepath2, dthresh):
    imprefixL = filepath1
    imprefixR = filepath2
    threshold = dthresh


    CLh,maskLh = decode(imprefixL,0,threshold)
    CLv,maskLv = decode(imprefixL,20,threshold)
    CRh,maskRh = decode(imprefixR,0,threshold)
    CRv,maskRv = decode(imprefixR,20,threshold)

    CL = CLh + 1024*CLv
    maskL = maskLh*maskLv
    CR = CRh + 1024*CRv
    maskR = maskRh*maskRv

    h = CR.shape[0]
    w = CR.shape[1]

    subR = np.nonzero(maskR.flatten())
    subL = np.nonzero(maskL.flatten())

    CRgood = CR.flatten()[subR]
    CLgood = CL.flatten()[subL]

    _,submatchR,submatchL = np.intersect1d(CRgood,CLgood,return_indices=True)

    matchR = subR[0][submatchR]
    matchL = subL[0][submatchL]

    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))

    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)
    
    pts3 = cu.triangulate(pts2L,camL,pts2R,camR)
    
    good = np.nonzero((pts3[2,:]>-200)&(pts3[2,:]<200))
    visutils.vis_scene(camL,camR,pts3[:,good[0]],looklength=5)
    
    return pts2L, pts2R, pts3

def m(pts2L, pts2R, pts3, blim, trithresh, color1, color2, plyname):
    pts3 = cu.triangulate(pts2L,camL,pts2R,camR)

    goodpts = np.nonzero((pts3[0,:]>blim[0])&(pts3[0,:]<blim[1]) & \
                      (pts3[1,:]>blim[2])&(pts3[1,:]<blim[3])& \
                      (pts3[2,:]>blim[4])&(pts3[2,:]<blim[5])) 

    pts3 = pts3[:,goodpts[0]]
    pts2L = pts2L[:,goodpts[0]]
    pts2R = pts2R[:,goodpts[0]]


    Triangles = Delaunay(pts2L.T)
    tri = Triangles.simplices

    d01 = np.sqrt(np.sum(np.power(pts3[:,tri[:,0]]-pts3[:,tri[:,1]],2),axis=0))
    d02 = np.sqrt(np.sum(np.power(pts3[:,tri[:,0]]-pts3[:,tri[:,2]],2),axis=0))
    d12 = np.sqrt(np.sum(np.power(pts3[:,tri[:,1]]-pts3[:,tri[:,2]],2),axis=0))

    goodtri = (d01<trithresh)&(d02<trithresh)&(d12<trithresh)

    tri = tri[goodtri,:]


    tokeep = np.unique(tri)
    remap = np.zeros(pts3.shape[1],dtype='int')
    remap[tokeep]= np.arange(0,tokeep.shape[0])
    pts3 = pts3[:,tokeep]
    tri = remap[tri]
 

    color = grab_color(color1, color2, pts2L, pts2R)
    writeply(pts3, color, tri, plyname)
    
    return tri, pts3, color

tri, pts3, color = m(pts2L, pts2R, pts3, np.array([-2,18.5,1,22,17,27]), 2.25, "teapot/grab_0_u/color_C0_01.png", "teapot/grab_0_u/color_C1_01.png", "grab0.ply")
mesh = trimesh.Trimesh(vertices=pts3.T,faces=tri[:,[0,2,1]])
mesh.show(smooth=True)
pts2L, pts2R, pts3 = points3d("teapot/grab_1_u/frame_C0_", "teapot/grab_1_u/frame_C1_", 0.025)

tri, pts3, color = m(pts2L, pts2R, pts3, np.array([-1,19,6,22,17,28]), 1.5, "teapot/grab_1_u/color_C0_01.png", "teapot/grab_1_u/color_C1_01.png", "grab1.ply")
mesh = trimesh.Trimesh(vertices=pts3.T,faces=tri[:,[0,2,1]])
mesh.show(smooth=True)
pts2L, pts2R, pts3 = points3d("teapot/grab_2_u/frame_C0_", "teapot/grab_2_u/frame_C1_", 0.025)

tri, pts3, color = m(pts2L, pts2R, pts3, np.array([-2,18.5,1,22,17,27]), 2.25, "teapot/grab_2_u/color_C0_01.png", "teapot/grab_2_u/color_C1_01.png", "grab2.ply")
mesh = trimesh.Trimesh(vertices=pts3.T,faces=tri[:,[0,2,1]])
mesh.show(smooth=True)
pts2L, pts2R, pts3 = points3d("teapot/grab_3_u/frame_C0_", "teapot/grab_3_u/frame_C1_", 0.03)

tri, pts3, color = m(pts2L, pts2R, pts3, np.array([-2,18.5,1,22,17,27]), 1.5, "teapot/grab_3_u/color_C0_01.png", "teapot/grab_3_u/color_C1_01.png", "grab3.ply")
mesh = trimesh.Trimesh(vertices=pts3.T,faces=tri[:,[0,2,1]])
mesh.show(smooth=True)
pts2L, pts2R, pts3 = points3d("teapot/grab_4_u/frame_C0_", "teapot/grab_4_u/frame_C1_", 0.025)

tri, pts3, color = m(pts2L, pts2R, pts3, np.array([-2,18.5,1,22,15,27]), 1.5, "teapot/grab_4_u/color_C0_01.png", "teapot/grab_4_u/color_C1_01.png", "grab4.ply")
mesh = trimesh.Trimesh(vertices=pts3.T,faces=tri[:,[0,2,1]])
mesh.show(smooth=True)
pts2L, pts2R, pts3 = points3d("teapot/grab_5_u/frame_C0_", "teapot/grab_5_u/frame_C1_", 0.03)

tri, pts3, color = m(pts2L, pts2R, pts3, np.array([-2,18.5,1,22,15,25]), 1.5, "teapot/grab_5_u/color_C0_01.png", "teapot/grab_5_u/color_C1_01.png", "grab5.ply")
mesh = trimesh.Trimesh(vertices=pts3.T,faces=tri[:,[0,2,1]])
mesh.show(smooth=True)
pts2L, pts2R, pts3 = points3d("teapot/grab_5_u/frame_C0_", "teapot/grab_5_u/frame_C1_", 0.025)

tri, pts3, color = m(pts2L, pts2R, pts3, np.array([-2,18.5,1,22,15,25]), 1.5, "teapot/grab_6_u/color_C0_01.png", "teapot/grab_6_u/color_C1_01.png", "grab6.ply")
mesh = trimesh.Trimesh(vertices=pts3.T,faces=tri[:,[0,2,1]])
mesh.show(smooth=True)
pts2L, pts2R, pts3 = points3d("teapot/grab_6_u/frame_C0_", "teapot/grab_6_u/frame_C1_", 0.025)
