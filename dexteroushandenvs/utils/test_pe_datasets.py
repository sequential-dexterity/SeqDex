import cv2
import numpy as np# Load the normalized depth image
import torch
import open3d as o3d
import scipy.io as scio
import pickle
import h5py

def project_point_cloud(point_cloud, rgb_image):
    global id

    point_cloud = point_cloud.detach().cpu().numpy()
    rotation_matrix = np.array([[0, -1, 0],
                                [0, 0, -1],
                                [1, 0, 0]])
    inverse_rotation_matrix = rotation_matrix.T
    point_cloud = point_cloud @ inverse_rotation_matrix

    intrinsic_matrix = np.float32([[434.21990966796875,   0. ,        326.772],
                                [  0.  ,       433.04193115234375, 245.07],
                                [  0.  ,         0.        ,   1.        ]])

    rgb = rgb_image.permute(1, 2, 0)

    # rgb = rgb.detach().cpu().numpy() * 255.
    # rgb = rgb.astype(np.uint8)
    rgb_image = cv2.resize(rgb, (640, 480))

    print(rgb_image.shape)

    image_points = (point_cloud[:, :2] / point_cloud[:, 2:]) * intrinsic_matrix[:2, :2].diagonal()
    image_points += intrinsic_matrix[:2, 2]
    image_points = np.round(image_points).astype(int)

    for point in image_points:
        u, v = point

        # Check if the pixel coordinates are within the image boundaries
        if 0 <= u < rgb_image.shape[1] and 0 <= v < rgb_image.shape[0]:
            # Copy the RGB values from the RGB image to the projected image
            cv2.circle(rgb_image, center=(u, v), radius=5, thickness=-1, color=(0, 0, 255))

    id += 1

    return rgb_image


def compute_camera_intrinsics_matrix(image_width, image_heigth, horizontal_fov):
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov *= np.pi / 180

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

    K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    return K

env_id = 3
steps_id = 5

is_load_from_hdf5 = False

f = open('./vinv.pkl', "rb")
vinv = pickle.load(f)# Inverse the normalization to recover the original depth image
f = open('./proj.pkl', "rb")
proj = pickle.load(f)# Inverse the normalization to recover the original depth image

# depth = np.load('./_output_headless/data/3-0-depth.npy')# Inverse the normalization to recover the original depth image
f = open('./_output_headless/data/{}-{}-depth.pkl'.format(steps_id, env_id), "rb")
depth = pickle.load(f)# Inverse the normalization to recover the original depth image

seg_original = cv2.imread('./_output_headless/data/{}-{}-label.jpg'.format(steps_id, env_id))
seg_original *= 255

# cv2.imshow("label", seg_original)
# cv2.waitKey(0)

depth_buffer = depth
data = scio.loadmat("./_output_headless/data/{}-{}-meta.mat".format(steps_id, env_id))
o3d_transformer = data["ext_poses"]
transformer = data["poses"]

print(transformer[:3, 3])

origin_image = cv2.imread('./_output_headless/data/{}-{}-color.jpg'.format(steps_id, env_id))

if is_load_from_hdf5:
    with h5py.File('./_output_headless/lego_pose_datasets.hdf5', "r") as f:
        depth_buffer = f["data"]["demo_10"]["depth_image"][:]
        o3d_transformer = f["data"]["demo_10"]["ext_poses"][:]
        transformer = f["data"]["demo_10"]["poses"][:]
        origin_image = f = f["data"]["demo_10"]["color_image"][:]
        # cv2.imshow("label", f["data"]["demo_10"]["color_image"][:])
        # cv2.imshow("seg", f["data"]["demo_10"]["label_image"][:])
        # cv2.waitKey(0)

gt_pcd = o3d.io.read_point_cloud('/home/jmji/IsaacSimRenderer/source/tools/uniform_points.xyz', format='xyz')
xyz = np.asarray(gt_pcd.points) * 9

camera_matrix = compute_camera_intrinsics_matrix(640, 480, 65)
# print(xyz[:, 0].mean())
# print(xyz[:, 1].mean())
# print(xyz[:, 2].mean())
print(camera_matrix)

rotation_mat = transformer[:3, 0:3]
translation_vec = transformer[:3, 3]
rotation_vec, _ = cv2.Rodrigues(rotation_mat)

projected_head_pose_box_points, _ = cv2.projectPoints(xyz,
                                                rotation_vec,
                                                translation_vec,
                                                camera_matrix,
                                                distCoeffs=None)

WIDTH = 480
HEIGHT = 640
image = np.ones((WIDTH, HEIGHT, 3), dtype=np.uint8) * 255
# origin_image = f["data"]["demo_10"]["color_image"][:]
image[:, :, 0] = origin_image[:, :, 0]
image[:, :, 1] = origin_image[:, :, 1]
image[:, :, 2] = origin_image[:, :, 2]

for point in projected_head_pose_box_points:
    i = round(point[0, 0])
    j = round(point[0, 1])
    # print(point)
    if i < HEIGHT and j < WIDTH:
        # image[j, i, :] = 0
        # print(point)
        cv2.circle(image, center=(i, j), radius=5, thickness=-1, color=(0, 0, 255))

# cv2.imshow("lab", image)
# cv2.waitKey(0)

# exit()

fu = 2/proj[0, 0]
fv = 2/proj[1, 1]

# Ignore any points which originate from ground plane or empty space
centerU = 640/2
centerV = 480/2

ori_points = []
for i in range(640):
    for j in range(480):
        if depth_buffer[j, i] < -1:
            continue
        u = -(i-centerU)/(640)  # image-space coordinate
        v = (j-centerV)/(480)  # image-space coordinate
        d = depth_buffer[j, i]  # depth buffer value
        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
        p2 = X2*vinv  # Inverse camera view to get world coordinates
        ori_points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

o3d_pc = o3d.geometry.PointCloud()
o3d_pc.points = o3d.utility.Vector3dVector(ori_points)

o3d_pc = o3d_pc.transform(o3d_transformer)

gt_pcd = o3d.io.read_point_cloud('/home/jmji/IsaacSimRenderer/source/tools/uniform_points.xyz', format='xyz')
xyz = np.asarray(gt_pcd.points)

gt = o3d.geometry.PointCloud()
gt.points = o3d.utility.Vector3dVector(xyz)
gt.paint_uniform_color([1, 0, 0])
gt_t = gt.transform(transformer)

o3d.visualization.draw_geometries([o3d_pc, gt_t])