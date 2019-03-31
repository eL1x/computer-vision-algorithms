import cv2
import numpy as np
import time


def init_bg_model(frame, N=20):
    height, width = frame.shape
    allowed_neighbors = [-1, 0, 1]
    bg_model = np.zeros((height, width, N))

    for row in range(height):
        for col in range(width):
            for sample_num in range(N):
                neighbor_row = -1
                while neighbor_row > height-1 or neighbor_row < 0:
                    row_shift = np.random.choice(allowed_neighbors)
                    neighbor_row = row + row_shift

                neighbor_col = -1
                while neighbor_col > width-1 or neighbor_col < 0:
                    col_shift = np.random.choice(allowed_neighbors)
                    neighbor_col = col + col_shift

                bg_model[row,col,sample_num] = frame[neighbor_row,neighbor_col]

    return bg_model


def vibe(frame, bg_model, N=20, R=30, num_min=2, fi=16):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if bg_model is None:
        bg_model = init_bg_model(frame, N)

    height, width = frame.shape
    seg_map = np.zeros((height,width)).astype(np.uint8)
    foreground = 255
    background = 0
    allowed_neighbors = [-1, 1]

    start_time = time.time()
    for row in range(height):
        for col in range(width):
            counter = 0
            for sample_num in range(N):
                dist = np.abs(frame[row,col] - bg_model[row,col,sample_num])
                if dist < R:
                    counter += 1

                if counter >= num_min:
                    break
            
            if counter >= num_min:
                seg_map[row,col] = background
                
                rand = np.random.randint(0, fi-1)
                if rand == 0:
                    sample_to_update = np.random.randint(0, N-1)
                    bg_model[row,col,sample_to_update] = frame[row,col]

                rand = np.random.randint(0, fi-1)
                if rand == 0:
                    neighbor_row = -1
                    while neighbor_row > height-1 or neighbor_row < 0:
                        row_shift = np.random.choice(allowed_neighbors)
                        neighbor_row = row + row_shift

                    neighbor_col = -1
                    while neighbor_col > width-1 or neighbor_col < 0:
                        col_shift = np.random.choice(allowed_neighbors)
                        neighbor_col = col + col_shift

                    sample_to_update = np.random.randint(0, N-1)
                    bg_model[neighbor_row,neighbor_col,sample_to_update] = frame[row,col]

            else:
                seg_map[row,col] = foreground

    end_time = time.time()
    print('{0} seconds'.format(end_time-start_time))
    
    return seg_map, bg_model
    

def main():
    with open('pedestrians/temporalROI.txt', 'r') as roi_info:
        line = roi_info.read()
        roi_start, roi_end = line.split()
        roi_start = int(roi_start)
        roi_end = int(roi_end)

    bg_model = None
    for i in range(roi_start, roi_end, 1):
        frame = cv2.imread('pedestrians/input/in%06d.jpg' % i)
        seg_map, bg_model = vibe(frame, bg_model)

        cv2.imshow("Image", np.uint8(frame))
        cv2.imshow("Seg_map", np.uint8(seg_map))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

