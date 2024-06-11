from flask import render_template
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2
from ultralytics import YOLO

def grayscale():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    new_arr = r.astype(int) + g.astype(int) + b.astype(int)
    new_arr = (new_arr/3).astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True


def zoomin():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def zoomout():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x/2)):
        for j in range(0, int(y/2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    new_arr = np.uint8(new_arr)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_left():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    h_img, w_img, _ = img.shape
    out = np.zeros((h_img-2, w_img-2), dtype=np.float)
    new_img = np.zeros((h_img-2, w_img-2, 3))
    if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]) == True:
        array = img[:, :, 0]
        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        for channel in range(3):
            new_img[:, :, channel] = out_
    else:
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
    new_img = np.uint8(new_img)
    return new_img


def edge_detection():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def blur():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :, 0].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)
        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv2.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/img/img_now.jpg', image_equalized)


def threshold(lower_thres, upper_thres):
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    _, thresholded_img = cv2.threshold(img, lower_thres, upper_thres, cv2.THRESH_BINARY)
    cv2.imwrite("static/img/img_now.jpg", thresholded_img)


def dilate_image():
    img = cv2.imread("static/img/img_now.jpg")
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite("static/img/img_now.jpg", dilated_img)

def erode_image():
    img = cv2.imread("static/img/img_now.jpg")
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite("static/img/img_now.jpg", eroded_img)

def opening_image():
    img = cv2.imread("static/img/img_now.jpg")
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=1)
    eroded_img = cv2.dilate(eroded_img, kernel, iterations=1)
    cv2.imwrite("static/img/img_now.jpg", eroded_img)


def closing_image():
    img = cv2.imread("static/img/img_now.jpg")
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.dilate(img, kernel, iterations=1)
    eroded_img = cv2.erode(eroded_img, kernel, iterations=1)
    cv2.imwrite("static/img/img_now.jpg", eroded_img)

def counting_image():
    # Baca gambar dan konversi ke skala abu-abu
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    img_normal = cv2.imread("static/img/img_normal.jpg")

    # Lakukan thresholding untuk menghasilkan gambar biner
    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Temukan kontur dalam gambar biner
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar kontur pada gambar normal
    img_with_contours = img_normal.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    # Hitung jumlah blob
    num_blobs = len(contours)

    # Simpan gambar dengan kontur
    cv2.imwrite("static/img/img_now.jpg", img_with_contours)

    # Kembalikan jumlah blob jika diperlukan
    return num_blobs


def extract_contour(image):
    if image is None or image.size == 0:
        print("Error: Gambar tidak berhasil dimuat atau kosong.")
        return []
    resized_image = cv2.resize(image, (100, 100))
    inverted_image = cv2.bitwise_not(resized_image)
    _, inverted_image = cv2.threshold(inverted_image, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    chain_code = []
    for i in range(1, len(contour)):
        dx = contour[i][0][0] - contour[i-1][0][0]
        dy = contour[i][0][1] - contour[i-1][0][1]
        direction = (dy > 0) * 3 + (dy < 0) * 7 + (dx > 0) * 1 + (dx < 0) * 5
        chain_code.append(direction)
    
    return chain_code

def extract_contours(image, min_contour_area=120):
    if image is None or image.size == 0:
        print("Error: Gambar tidak berhasil dimuat atau kosong.")
        return []
    resized_image = cv2.resize(image, (100, 100))
    inverted_image = cv2.bitwise_not(resized_image)
    _, inverted_image = cv2.threshold(inverted_image, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    
    chain_codes = []
    for contour in contours:
        image_with_contours = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.drawContours(image_with_contours, contour, -1, (0, 255,0), 2)
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            chain_code = []
            for i in range(1, len(contour)):
                dx = contour[i][0][0] - contour[i-1][0][0]
                dy = contour[i][0][1] - contour[i-1][0][1]
                direction = (dy > 0) * 3 + (dy < 0) * 7 + (dx > 0) * 1 + (dx < 0) * 5
                chain_code.append(direction)
            chain_codes.append(chain_code)
    return chain_codes

def read_from_txt(filename):
    chain_codes = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            digit = int(lines[i].split(':')[1])
            code = list(map(int, lines[i+1].split(',')))
            chain_codes[digit] = code
            i += 2
            while i < len(lines) and not lines[i].strip():
                i += 1
    return chain_codes

def hamming_distance(code1, code2):
    len1 = len(code1)
    len2 = len(code2)
    min_len = min(len1, len2)
    distance = 0
    for i in range(min_len):
        if code1[i] != code2[i]:
            distance += 1
    distance += abs(len1 - len2)
    return distance


def multi_digit_detection(image, knowledge_base):
    chain_codes = extract_contours(image)
    detected_digits = []
    for chain_code in chain_codes:
        min_distance = float('inf')
        detected_digit = None
        for digit, code in knowledge_base.items():
            distance = hamming_distance(chain_code, code)
            if distance < min_distance:
                min_distance = distance
                detected_digit = digit
        detected_digits.append(detected_digit)
    return detected_digits[::-1]

def checkdigit():
    knowledge_base = read_from_txt('sampleKontur/knowledge_base.txt')
    test_image_path = 'static/img/img_normal.jpg'
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    detected_digit = multi_digit_detection(test_image, knowledge_base)
    detected_digits_str = ''.join(map(str, detected_digit))
    return detected_digits_str

def extract_contour(image):
    if image is None or image.size == 0:
        print("Error: Gambar tidak berhasil dimuat atau kosong.")
        return []

    resized_image = cv2.resize(image, (100, 100))
    inverted_image = cv2.bitwise_not(resized_image)
    _, inverted_image = cv2.threshold(inverted_image, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return []
    
    contour = max(contours, key=cv2.contourArea)
    chain_code = []
    for i in range(1, len(contour)):
        dx = contour[i][0][0] - contour[i-1][0][0]
        dy = contour[i][0][1] - contour[i-1][0][1]
        direction = (dy > 0) * 3 + (dy < 0) * 7 + (dx > 0) * 1 + (dx < 0) * 5
        chain_code.append(direction)
    
    return chain_code

def extract_contours(image, min_contour_area=120):
    if image is None or image.size == 0:
        print("Error: Gambar tidak berhasil dimuat atau kosong.")
        return []

    resized_image = cv2.resize(image, (100, 100))
    inverted_image = cv2.bitwise_not(resized_image)
    _, inverted_image = cv2.threshold(inverted_image, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return []
    
    chain_codes = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            chain_code = []
            for i in range(1, len(contour)):
                dx = contour[i][0][0] - contour[i-1][0][0]
                dy = contour[i][0][1] - contour[i-1][0][1]
                direction = (dy > 0) * 3 + (dy < 0) * 7 + (dx > 0) * 1 + (dx < 0) * 5
                chain_code.append(direction)
            chain_codes.append(chain_code)
    return chain_codes

def save_to_txt(chain_codes, filename):
    with open(filename, 'w') as file:
        for emoji, code in chain_codes.items():
            file.write(f"Emoji: {emoji}\n")
            file.write(','.join(map(str, code)) + '\n')

def read_from_txt(filename):
    chain_codes = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if not lines[i].strip():  # Skip empty lines
                i += 1
                continue
            emoji = lines[i].split(':')[1].strip()
            code_line = lines[i+1].strip()
            if code_line:  # Ensure the code line is not empty
                code = list(map(int, code_line.split(',')))
                chain_codes[emoji] = code
            i += 2
    return chain_codes

def hamming_distance(code1, code2):
    min_len = min(len(code1), len(code2))
    distance = sum(c1 != c2 for c1, c2 in zip(code1[:min_len], code2[:min_len]))
    distance += abs(len(code1) - len(code2))
    return distance

def multi_emoji_detection(image, knowledge_base):
    chain_codes = extract_contours(image)
    
    detected_emojis = []
    for chain_code in chain_codes:
        min_distance = float('inf')
        detected_emoji = None
        for emoji, code in knowledge_base.items():
            distance = hamming_distance(chain_code, code)
            if distance < min_distance:
                min_distance = distance
                detected_emoji = emoji
        detected_emojis.append(detected_emoji)
    
    return detected_emojis[::-1]

def checkemoji():
    # knowledge_base = {}
    # emojis = ["rolling_on_the_floor_laughing","sleeping", "sleepy", "slightly_smiling_face","smile", "smiley", "smirk", 
    #           "star-struck", "sunglasses", "sweat_smile", "thinking_face", "tired_face", "wink","yum", "zipper_mouth_face",
    #           "blush", "disappointed_relieved", "expressionless", "face_with_raised_eyebrow", "face_with_rolling_eyes", "grin", "grinning", "heart_eyes",
    #           "hugging_face", "hushed", "joy", "kissing", "kissing_closed_eyes", "kissing_heart", "kissing_smiling_eyes", "laughing", "neutral_face",
    #           "no_mouth", "open_mouth", "persevere", "relaxed"]
    # for emoji in emojis:
    #     emoji_image_path = f"emoji/{emoji}.png"
    #     emoji_image = cv2.imread(emoji_image_path, cv2.IMREAD_GRAYSCALE)       
    #     emoji_chain_code = extract_contour(emoji_image)
    #     knowledge_base[emoji] = emoji_chain_code

    # # save_to_txt(knowledge_base, 'emoji_knowledge_base.txt')
    knowledge_base = read_from_txt('sampleKontur/emoji_knowledge_base.txt')

    test_image_path = 'static/img/img_normal.jpg'
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    detected_emojis = multi_emoji_detection(test_image, knowledge_base)
    detected_emojis_str = ', '.join(map(str, detected_emojis))
    return detected_emojis_str

def mask_detect():
    # Load the trained model
    model = YOLO('SampleYolo/best.pt')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set the initial confidence threshold
    min_conf_threshold = 0.8

    def on_trackbar(val):
        nonlocal min_conf_threshold
        min_conf_threshold = val / 100

    # Create a window and a trackbar for adjusting the threshold
    cv2.namedWindow('YOLOv8 Live Prediction')
    cv2.createTrackbar('Confidence Threshold', 'YOLOv8 Live Prediction', int(min_conf_threshold * 100), 100, on_trackbar)

    # Define colors for "mask" and "no-mask"
    color_mask = (0, 255, 0)  # Green for mask
    color_no_mask = (0, 0, 255)  # Red for no mask

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run prediction on the frame
        results = model.predict(source=frame)

        # Draw the results on the frame
        for result in results:
            for box in result.boxes:
                conf = box.conf[0]
                if conf >= min_conf_threshold:  # Apply the confidence threshold
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = box.cls[0]
                    label = f'{model.names[int(cls)]} {conf:.2f}'

                    # Choose color based on the class label
                    if model.names[int(cls)] == 'mask':
                        color = color_mask
                    else:
                        color = color_no_mask

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield the output frame in byte format
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


def mask_image_detection(image_path, output_path):
    # Load the trained model
    model = YOLO('SampleYolo/best.pt')

    # Load the image from file
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    # Convert image to RGB format (PIL uses RGB)
    image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_pil)

    # Set the confidence threshold
    min_conf_threshold = 0.8

    # Define colors for "mask" and "no-mask"
    color_mask = (0, 255, 0)  # Green for mask
    color_no_mask = (0, 0, 255)  # Red for no mask

    # Run prediction on the image
    results = model.predict(source=image)

    # Draw the results on the image
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    for result in results:
        for box in result.boxes:
            conf = box.conf[0]
            if conf >= min_conf_threshold:  # Apply the confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = box.cls[0]
                label = f'{model.names[int(cls)]} {conf:.2f}'

                # Choose color based on the class label
                if model.names[int(cls)] == 'mask':
                    color = color_mask
                else:
                    color = color_no_mask

                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Set font size to a larger value
                font_size = 20
                font = ImageFont.truetype("arial.ttf", font_size)
                
                draw.text((x1, y1 - 20), label, font=font, fill=color)

    # Save the processed image to the specified output path
    image_pil.save(output_path)
    print(f"Processed image saved to {output_path}")