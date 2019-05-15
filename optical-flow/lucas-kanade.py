import cv2
import numpy as np
import matplotlib.pyplot as plt


def lucas_kanade(image1, image2):
    # Init image flow
    u = np.zeros(image1.size)
    v = np.zeros(image1.size)

    Ix, Iy, It = compute_derivatives(image1, image2)

    # Compute elements needed to build A.T*A
    Ix2 = Ix**2
    Iy2 = Iy**2
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It

    # Compute sum of elements in window
    window_size = 17
    Ix2_sum = compute_window_sum(Ix2, window_size)
    Iy2_sum = compute_window_sum(Iy2, window_size)
    IxIy_sum = compute_window_sum(IxIy, window_size)
    IxIt_sum = compute_window_sum(IxIt, window_size)
    IyIt_sum = compute_window_sum(IyIt, window_size)

    # Solve equation v = (A.T*A)^(-1) * A.T*b
    det = Ix2_sum * Iy2_sum - IxIy_sum**2
    det[np.where(det == 0)] = 1 # To prevent division by zero
    # print(det.min())
    u = IxIy_sum * IyIt_sum - Iy2_sum * IxIt_sum
    v = IxIt_sum * IxIy_sum - Ix2_sum * IyIt_sum
    u = u / det
    v = v / det

    # print(u.max())
    return u, v


def compute_derivatives(image1, image2):
    # Masks needed to compute Ix, Iy, It
    Ix_mask = np.array([[-1, 1], [-1, 1]]) * 0.25
    Iy_mask = np.array([[-1, -1], [1, 1]]) * 0.25
    It_mask = np.array([[1, 1], [1, 1]]) * 0.25

    # Compute Ix, Iy, It
    Ix = cv2.filter2D(image1, cv2.CV_64F, Ix_mask) + cv2.filter2D(image2, cv2.CV_64F, Ix_mask)
    Iy = cv2.filter2D(image1, cv2.CV_64F, Iy_mask) + cv2.filter2D(image2, cv2.CV_64F, Iy_mask) 
    It = cv2.filter2D(image1, cv2.CV_64F, It_mask) + cv2.filter2D(image2, cv2.CV_64F, -It_mask)

    return Ix, Iy, It


def compute_window_sum(matrix, window_size=3):
    mask = np.ones((window_size, window_size))
    window_sum = cv2.filter2D(matrix, cv2.CV_64F, mask)

    return window_sum


def plot_solution(image, u, v, scale=1):
    mesh_padding = 7
    plt.figure()
    plt.imshow(image, cmap='gray')
    for i in range(0, u.shape[0], mesh_padding):
            for j in range(0, u.shape[1], mesh_padding):
                    plt.arrow(j, i, v[i,j]*scale, u[i,j]*scale, color='red', head_width=1)


def main():
     # Load images in grayscale
    image1_orig = cv2.imread('images/image1.jpg', 0)
    image2_orig = cv2.imread('images/image2.jpg', 0)
    
    # Use gaussian blur to reduce noise
    mask = (51, 51)
    image1 = cv2.GaussianBlur(image1_orig, mask, 0)
    image2 = cv2.GaussianBlur(image2_orig, mask, 0)

    u, v = lucas_kanade(image1, image2)

    # Plot 
    plot_solution(image2_orig, u, v)
    plt.show()


def draw_arrows(image, u, v):
    mesh_padding = 7
    for i in range(0, u.shape[0], mesh_padding):
            for j in range(0, u.shape[1], mesh_padding):
                    cv2.arrowedLine(image, (i, j), (i + int(u[i,j]), j + int(v[i,j])), (0,0,255))

    return image

def feed_from_cam():
    mask = (51, 51)

    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, mask, 0)

    while (True):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, mask, 0)

        u, v = lucas_kanade(prev_frame, frame_gray)

        frame_w_flow = draw_arrows(frame_gray, u, v)

        # Display the resulting frame
        cv2.imshow('frame', frame_w_flow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame_gray.copy()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    feed_from_cam()