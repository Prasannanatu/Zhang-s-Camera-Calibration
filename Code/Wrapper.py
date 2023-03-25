#!/usr/bin/env python3

import numpy as np
import cv2 
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import math
import os,sys
import pry
from scipy.linalg import svd
import scipy.optimize
import glob

def get_H(save_directory, images_list, world_points):
    
    copy_images = np.copy(images_list)
    H_matrices = []
    image_points = []

    for i, image in enumerate(copy_images):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners_list = cv2.findChessboardCorners(grayscale, (9, 6), None)
        if ret: 
            corners = corners_list.reshape(-1, 2)
            H_matrix, _ = cv2.findHomography(world_points, corners, cv2.RANSAC, 5.0)
            H_matrices.append(H_matrix)
            image_points.append(corners)
            cv2.drawChessboardCorners(image, (9, 6), corners, True)
            image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
            cv2.imwrite(save_directory + '/' + str(i) + '_corners.png', image)
    return H_matrices, image_points


def v(i, j, H):
    return np.array([
        H[0][i] * H[0][j],
        H[0][i] * H[1][j] + H[1][i] * H[0][j],
        H[1][i] * H[1][j],
        H[2][i] * H[0][j] + H[0][i] * H[2][j],
        H[2][i] * H[1][j] + H[1][i] * H[2][j],
        H[2,][i] * H[2][j]
    ])


def get_K(H_total):
    V= []
    for h in H_total:
        V.append(v(0, 1, h))
        V.append(v(0, 0, h) - v(1, 1, h))
    V = np.array(V)
    U, sigma, V_t = np.linalg.svd(V)# scipy.linalg.svd doesnt work here don't know why??
    # print(V_t.shape)
    b = V_t[-1][:]
    # print(b.shape)
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]
    
    B = np.array([[B11,B12 ,B13],
                 [B12 ,B22, B23],
                 [B13,B23, B33]])
    
    vo = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    lambda_ = B33 - (B13**2 + vo*(B12*B13 - B11*B23))/B11
    print("this is B11", B11)
    aplha =  np.sqrt(lambda_/B11)
    beta = math .sqrt((lambda_ * B11)/(B11*B22 - B12**2))
    gamma_ = - (B12 * aplha *aplha *beta) / lambda_
    u_o  = ((gamma_ * vo) / beta) - ((B13 * aplha * aplha) / lambda_)


    # vo = [( B[0][1]* B[0][2] ) - ( B[0][0] * B[1][2]) ] / ((B[0][0]*B[1][1]) - (B[0][1] * B[0][1]))
    # lambda_ = B[1][1] - ([(B[0][2] * B[0][2]) + (vo *(B[0][1]* B[0][2]) - (B[0][0] * B[1][2]))]/ B[0][0])
    # aplha = math.sqrt((lambda_ / B[0][0]))
    # beta = math .sqrt((lambda_ * B[0][0])/((B[0][0] *B[1][1]) - (B[0][1] *B[0][1])))
    # gamma_ = - (B[0][1] * aplha *aplha *beta) / lambda_
    # u_o  = ((gamma_ * vo) / beta) - ((B[0][2] * aplha * aplha) / lambda_)
    # 
    K = np.array([[aplha , gamma_, u_o],
                  [0 , beta , vo],
                  [0,0,1]])


    return K

def get_R(K, H_total):
    R = []
    k_inv = np.linalg.pinv(K)

    for h in H_total:
        h1 =h[:,0]
        h2 =h[:,1]
        h3 =h[:,2]
        lamda_e = 1/scipy.linalg.norm((k_inv @ h1), 2)
        r1 = lamda_e * (k_inv @ h1) 
        # print(r1)
        r2 = lamda_e * (k_inv @ h2) 
        # print(r2)
        # r3 = np.cross(r1, r2)
        # print(r3)
        t = lamda_e * (k_inv @ h3)
        rt = np.vstack((r1, r2, t)).T
        R.append(rt)
    return R


def get_loss_function(K ,corner_list, world_coordinates, R):
    alpha, gamma, beta, uo, vo, k1, k2 = K 
    K = np.array([[alpha , gamma, uo],
                  [0 , beta , vo],
                  [k1,k2,1]])
    f_error = []
    
    for i, corner_list in enumerate(corner_list):
        K_ = (K @ R[i].reshape(3,3))

        total_error=0

        for j in range(len(world_coordinates)):

            world_point = world_coordinates[j]
            # print("world point", world_point[0])
            # print("shape of world point", world_point.shape)
            M = np.array([world_point[0], world_point[1], 1]).reshape(3, 1)

            proj_pts = (R[i] @ M) #Equation 1
            proj_pts /= proj_pts[2]
            x, y = proj_pts[0], proj_pts[1]

            projected_coordinates = (K_ @ M)
            # print("size of H", H.shape)
            # print("size of H", world_points.shape)
            u = projected_coordinates[0] / projected_coordinates[2]
            v = projected_coordinates[1] / projected_coordinates[2]


            # print("size of H", H.shape)
            # print("size of r", R.shape)

            corner_points = corner_list[j]
            corner_points = np.array([corner_points[0], corner_points[1], 1], dtype = np.float32)

            # r = x**2 + y**2
            # print(x)
            u_hat = u + (u-uo)*(k1*(x**2 + y**2) + k2*((x**2 + y**2)**2))
            # print("u_hat", u_hat)
            v_hat = v + (v-vo)*(k1*(x**2 + y**2 )+ k2*((x**2 + y**2)**2))
            corrected_corner = np.array([u_hat, v_hat, 1], dtype = np.float32)

            error = np.linalg.norm(corner_points - corrected_corner, 2)
            # print ("error", error)
            total_error = total_error + error
        
        f_error.append(total_error/54)
        # print("reporjected",reporjected)
        # proj_points.append(projected_corners)
    # mean_error = np.sum(np.array(reporjected)) / (len(corner_list) * world_coordinates.shape[0])
    # print("error", f_error)

    return np.array(f_error)


def get_optimised(K, corner_list ,world_coordinates , R):
    alpha = K[0, 0]
    gamma = K[0, 1]
    uo = K[0, 2]
    beta = K[1, 1]
    vo = K[1, 2]
    k1 = K[2, 0]
    k2 = K[2, 1]
    optimized = scipy.optimize.least_squares(fun=get_loss_function, x0 = [alpha, gamma, beta, uo, vo, k1, k2], method = 'lm', args=(corner_list, world_coordinates, R))
    [alpha_u, gamma_u, beta_u, uo_u, vo_u ,k1_u, k2_u] = optimized.x
    updated_K = np.array([[alpha_u , gamma_u, uo_u],
                  [0, beta_u, vo_u],
                  [0, 0, 1]])
    return updated_K, k1_u, k2_u


def get_error(updated_K ,updated_K_distorted, corner_list, world_coordinates, R):
    uo = updated_K[0][2]
    vo = updated_K[1][2]
    # alpha, gamma, beta, uo, vo, k1, k2 = updated_K 
    # K = np.array([[alpha , gamma, uo],
    #               [0 , beta , vo],
    #               [0,0,1]])
    k1 = updated_K_distorted[0]
    k2 = updated_K_distorted[1]
    total_error = []
    proj_points = []
    reporjected = []
    for i,corner in enumerate(corner_list):
        K_ = updated_K @ R[i]

        error_sum = 0
        projected_corners = []
        for j in range(world_coordinates.shape[0]):
            world_point = world_coordinates[j]
            M = np.array([world_point[0], world_point[1], 1]).reshape(3, 1)

            image_coordinates = (R[i] @ M)
            x, y = image_coordinates[0]/image_coordinates[2], image_coordinates[1]/image_coordinates[2]

            projected_coordinates = (K_ @ M)
            u = projected_coordinates[0] / projected_coordinates[2]
            v = projected_coordinates[1] / projected_coordinates[2]

            corner_points = corner[j]
            corner_points = np.array([corner_points[0], corner_points[1], 1])

            u_hat = u+(u - uo)*(k1*(x**2 + y**2)+k2*(x**2 + y**2)**2)
            v_hat = v+(v - vo)*(k1*(x**2 + y**2)+k2*(x**2 + y**2)**2)
            
            projected_corners.append([int(u_hat), int(v_hat)])
            corrected_corner = np.array([u_hat, v_hat, 1])
            error = np.linalg.norm(corner_points - corrected_corner, ord =2)
            total_error += error

        reporjected.append(error_sum)
        proj_points.append(projected_corners)

    re_error_avg = np.sum(np.array(reporjected)) / (len(corner_list) * world_coordinates.shape[0])
    return re_error_avg, proj_points
    



def main():

    image_path = '/home/prasanna/Documents/courses/SEM-2/CV/Assignment/pvnatu_hw1/Calibration_Imgs/Calibration_Imgs/'
    Save_path = '/home/prasanna/Documents/courses/SEM-2/CV/Assignment/pvnatu_hw1/Calibration_Imgs/Outputs1/'
 
    Images = [cv2.imread(file) for file in glob.glob(image_path + '*.jpg')]# reading image from the image directory
    world_coordinates_x, world_coordinates_y = np.meshgrid(range(9), range(6))# the actual coordinates will be multiple of this size of box

    world_coordinates_x =world_coordinates_x.reshape(54, 1)# reshaping the coordinates
    world_coordinates_y =world_coordinates_y.reshape(54, 1)# reshaping the coordinates
    world_coordinates = np.array(np.hstack((world_coordinates_x, world_coordinates_y)).astype(np.float32)*21.5)# stacking all values converting to float and multiplying with actual value to get world coordinates

    H_total, corner_list = get_H(Save_path, Images, world_coordinates)# get the corners of checkerboard and get the homography between world points and image points also.
    K = get_K(H_total)
    # print("before_update K", K)
    R = get_R(K, H_total)
    # print("before_update R", R)

    K_updated, k1, k2 = get_optimised(K, corner_list, world_coordinates, R)# optimising the valeus of K and distortion parameters
    distortion_updated = np.array([k1, k2]).reshape(2, 1)
    R_updated = get_R(K_updated, H_total)# getting updated R

    reprojected_error, reprojected_points = get_error(K_updated, distortion_updated, corner_list, world_coordinates, R_updated) # recalculating error after optimization
    # print("first error",reprojected_error)
    
    K_final = np.array(K_updated, np.float32).reshape(3,3)
    print("K matrix: \n", K_final)
    distortion = np.array([distortion_updated[0],distortion_updated[1], 0, 0], np.float32)

    
    print('Distortion optimized: ', distortion_updated)
    print("Error: ", reprojected_error)

    for i,image_points in enumerate(reprojected_points):
        image = cv2.undistort(Images[i], K_final, distortion)
        for point in image_points:
            image = cv2.circle(image, (int(point[0]),int(point[1])), 5, (128, 0, 128), 6)
        # cv2.imshow('frame', image)
        filename =Save_path + str(i) + "reprojected_image.png"
        cv2.imwrite(filename, image)




if __name__ == '__main__':
    main()
