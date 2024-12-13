import numpy as np
import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from calibration import load_calibration
from copy import copy

class Lane():
    def __init__(self):
        self.detected = False
        self.cur_fitx = None
        self.cur_fity = None
        self.prev_fitx = []
        self.current_poly = [np.array([False])]
        self.prev_poly = [np.array([False])]

    def average_pre_lanes(self):
        tmp = copy(self.prev_fitx)
        tmp.append(self.cur_fitx)
        self.mean_fitx = np.mean(tmp, axis=0)

    def append_fitx(self):
        if len(self.prev_fitx) == N:
            self.prev_fitx.pop(0)
        self.prev_fitx.append(self.mean_fitx)

    def process(self, ploty):
        self.cur_fity = ploty
        self.average_pre_lanes()
        self.append_fitx()
        self.prev_poly = self.current_poly


left_lane = Lane()
right_lane = Lane()
frame_width = 1280
frame_height = 720

LANEWIDTH = 3.7
input_scale = 4
output_frame_scale = 4
N = 4

x = [194, 1117, 705, 575]
y = [719, 719, 461, 461]
X = [290, 990, 990, 290]
Y = [719, 719, 0, 0]

src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]]) / input_scale)
dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]]]) / input_scale)

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

X_b = [574, 706, 706, 574]
Y_b = [719, 719, 0, 0]
src_ = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]]) / (input_scale*2))
dst_ = np.floor(np.float32([[X_b[0], Y_b[0]], [X_b[1], Y_b[1]], [X_b[2], Y_b[2]], [X_b[3], Y_b[3]]]) / (input_scale*2))
M_b = cv2.getPerspectiveTransform(src_, dst_)

s_thresh, sx_thresh, dir_thresh, m_thresh, r_thresh = (120, 255), (20, 100), (0.7, 1.3), (30, 100), (200, 255)

calib_file = 'camera_cal/calibration_pickle.p'
mtx, dist = load_calibration(calib_file)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output



def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def gradient_pipeline(image, ksize = 3, sx_thresh=(20, 100), sy_thresh=(20, 100), m_thresh=(30, 100), dir_thresh=(0.7, 1.3)):

    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=sx_thresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=sy_thresh)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=m_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=dir_thresh)
    combined = np.zeros_like(mag_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def threshold_col_channel(channel, thresh):

    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary


def find_edges(img, s_thresh=s_thresh, sx_thresh=sx_thresh, dir_thresh=dir_thresh):
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    s_binary = threshold_col_channel(s_channel, thresh=s_thresh)
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)
    combined_binary = np.zeros_like(s_channel)
    combined_binary[((sxbinary == 1) & (dir_binary==1)) | ((s_binary == 1) & (dir_binary==1))] = 1
    c_bi = np.zeros_like(s_channel)
    c_bi[((sxbinary == 1) & (s_binary==1))] = 2
    ave_binary = (combined_binary + c_bi)
    return ave_binary


def warper(img, M):

    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def full_search(binary_warped, visualization=False):

    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.astype('uint8')

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = np.floor(100/input_scale)
    minpix = np.floor(50/input_scale)
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if visualization:
            cv2.rectangle(out_img,(int(win_xleft_low),int(win_y_low)),(int(win_xleft_high),int(win_y_high)),(0,255,0), 2)
            cv2.rectangle(out_img,(int(win_xright_low),int(win_y_low)),(int(win_xright_high),int(win_y_high)),(0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if visualization:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width / input_scale))
        plt.ylim((frame_height / input_scale, 0))
        plt.show()

    return left_fit, right_fit



def window_search(left_fit, right_fit, binary_warped, margin=100, visualization=False):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if visualization:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = out_img.astype('uint8')
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width / input_scale))
        plt.ylim((frame_height / input_scale, 0))

        plt.show()

    return left_fit, right_fit

def measure_lane_curvature(ploty, leftx, rightx, visualization=False):
    leftx = leftx[::-1]
    rightx = rightx[::-1]

    y_eval = np.max(ploty)

    ym_per_pix = 30 / (frame_height / input_scale)
    xm_per_pix = LANEWIDTH / (700 / input_scale)

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

    if leftx[0] - leftx[-1] > 50 / input_scale:
        curve_direction = 'Left curve'
    elif leftx[-1] - leftx[0] > 50 / input_scale:
        curve_direction = 'Right curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad + right_curverad) / 2.0, curve_direction


def off_center(left, mid, right):
    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  
        offset = a / width * LANEWIDTH - LANEWIDTH /2.0
    else:       
        offset = LANEWIDTH /2.0 - b / width * LANEWIDTH

    return offset


def compute_car_offcenter(ploty, left_fitx, right_fitx, undist):

    height = undist.shape[0]
    width = undist.shape[1]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    bottom_l = left_fitx[height-1]
    bottom_r = right_fitx[0]

    offcenter = off_center(bottom_l, width/2.0, bottom_r)

    return offcenter, pts


def create_output_frame(offcenter, pts, undist_ori, fps, curvature, curve_direction, binary_sub, threshold=0.6):

    undist_ori = cv2.resize(undist_ori, (0,0), fx=1/output_frame_scale, fy=1/output_frame_scale)
    w = undist_ori.shape[1]
    h = undist_ori.shape[0]

    undist_birdview = warper(cv2.resize(undist_ori, (0,0), fx=1/2, fy=1/2), M_b)

    color_warp = np.zeros_like(undist_ori).astype(np.uint8)

    whole_frame = np.zeros((h*2.5,w*2.34, 3), dtype=np.uint8)


    if abs(offcenter) > threshold:  
        cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0)) # red
    else: 
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))  # green

    newwarp = cv2.warpPerspective(color_warp, M_inv, (int(frame_width/input_scale), int(frame_height/input_scale)))


    newwarp_ = cv2.resize(newwarp,None, fx=input_scale/output_frame_scale, fy=input_scale/output_frame_scale, interpolation = cv2.INTER_LINEAR)

    output = cv2.addWeighted(undist_ori, 1, newwarp_, 0.3, 0)

    whole_frame[40:40+h, 20:20+w, :] = undist_ori
    whole_frame[40:40+h, 60+w:60+2*w, :] = output
    whole_frame[220+h/2:220+2*h/2, 20:20+w/2, :] = undist_birdview
    whole_frame[220+h/2:220+2*h/2, 40+w/2:40+w, 0] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)
    whole_frame[220+h/2:220+2*h/2, 40+w/2:40+w, 1] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)
    whole_frame[220+h/2:220+2*h/2, 40+w/2:40+w, 2] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if offcenter >= 0:
        offset = offcenter
        direction = 'Right'
    elif offcenter < 0:
        offset = -offcenter
        direction = 'Left'

    info_road = "Road Status"
    info_lane = "Lane info: {0}".format(curve_direction)
    info_cur = "Curvature {:6.1f} m".format(curvature)
    info_offset = "Off center: {0} {1:3.1f}m".format(direction, offset)
    info_framerate = "{0:4.1f} fps".format(fps)
    info_warning = "Warning: offcenter > 0.6m (use higher threshold in real life)"

    cv2.putText(whole_frame, "Departure Warning System with a Monocular Camera", (23,25), font, 0.8, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Origin", (22,70), font, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Augmented", (40+w+25,70), font, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Bird's View", (22+30,70+35+h), font, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Lanes", (22+225,70+35+h), font, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, info_road, (40+w+50,70+35+h), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_warning, (35+w,60+h), font, 0.4, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_lane, (40+w+50,70+35+40+h), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_cur, (40+w+50,70+35+80+h), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_offset, (40+w+50,70+35+120+h), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_framerate, (40+w+250,70), font, 0.6, (255,255,0), 1,cv2.LINE_AA)

    return whole_frame


def tracker(binary_sub, ploty, visualization=False):

    left_fit, right_fit = window_search(left_lane.prev_poly, right_lane.prev_poly, binary_sub, margin=100/input_scale, visualization=visualization)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 /input_scale):
        left_lane.detected = True
        right_lane.detected = True
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
    else:
        left_lane.detected = False
        right_lane.detected = False
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        left_lane.cur_fitx = left_lane.prev_fitx[-1]
        right_lane.cur_fitx = right_lane.prev_fitx[-1]


def detector(binary_sub, ploty, visualization=False):

    left_fit, right_fit = full_search(binary_sub, visualization=visualization)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 /input_scale):
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
        left_lane.detected = True
        right_lane.detected = True
    else:
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        if len(left_lane.prev_fitx) > 0:
            left_lane.cur_fitx = left_lane.prev_fitx[-1]
            right_lane.cur_fitx = right_lane.prev_fitx[-1]
        else:
            left_lane.cur_fitx = left_fitx
            right_lane.cur_fitx = right_fitx
        left_lane.detected = False
        right_lane.detected = False



def process_frame(img, visualization=False):

    start = timer()
    img_undist_ = cv2.undistort(img, mtx, dist, None, mtx)
    img_undist = cv2.resize(img_undist_, (0,0), fx=1/input_scale, fy=1/input_scale)

    img_binary = find_edges(img_undist)

    binary_warped = warper(img_binary, M)  

    binary_sub = np.zeros_like(binary_warped)
    binary_sub[:, int(150/input_scale):int(-80/input_scale)]  = binary_warped[:, int(150/input_scale):int(-80/input_scale)]

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    if left_lane.detected:  
        tracker(binary_sub, ploty, visualization)
    else:  
        detector(binary_sub, ploty, visualization)

    left_lane.process(ploty)
    right_lane.process(ploty)

    curvature, curve_direction = measure_lane_curvature(ploty, left_lane.mean_fitx, right_lane.mean_fitx)

    offcenter, pts = compute_car_offcenter(ploty, left_lane.mean_fitx, right_lane.mean_fitx, img_undist)

    end = timer()
    fps = 1.0 / (end - start)

    output = create_output_frame(offcenter, pts, img_undist_, fps, curvature, curve_direction, binary_sub)
    return output



