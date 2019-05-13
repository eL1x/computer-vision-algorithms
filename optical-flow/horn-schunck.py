import numpy as np
import matplotlib.pyplot as plt
import cv2


def horn_schunck(image1, image2, alpha, iterations=100):
    # Init velocities
    # Prev values stored for stopping condition
    u = np.zeros(image1.shape)
    v = np.zeros(image1.shape)
    u_prev = np.zeros(image1.shape)
    v_prev = np.zeros(image1.shape)

    Ix, Iy, It = compute_derivatives(image1, image2)
    plot_derivatives(Ix, Iy, It)
    
    for i in range(iterations):
        u_avg, v_avg = compute_averages(u, v)
        numerator = Ix*u_avg + Iy*v_avg + It
        denominator = alpha**2 + Ix**2 + Iy**2
        # print(i)
        temp = (numerator/denominator)
        u = u_avg - Ix*temp
        v = v_avg - Iy*temp

        # Stopping condition
        solution_change = np.mean((u - u_prev)**2 + (v - v_prev)**2)
        epsilon = 10**(-6)
        if i != 0 and solution_change < epsilon:
            break

        u_prev = u
        v_prev = v

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


def compute_averages(u, v):
    # Estimate local averages
    avg_mask = np.array([[1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]])
    u_avg = cv2.filter2D(u, cv2.CV_64F, avg_mask)
    v_avg = cv2.filter2D(v, cv2.CV_64F, avg_mask)

    return u_avg, v_avg


def plot_derivatives(Ix, Iy, It):
    fg, ax = plt.subplots(1, 3, figsize=(18, 5))
    for f, a, t in zip((Ix, Iy, It), ax, ('$f_x$', '$f_y$', '$f_t$')):
        h=a.imshow(f, cmap='bwr')
        a.set_title(t)
        fg.colorbar(h, ax=a)


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

    u, v = horn_schunck(image1, image2, 10, 1000)

    # Plot 
    plot_solution(image2_orig, u, v)
    plt.show()


if __name__ == "__main__":
    main()
