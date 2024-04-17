import cv2
import numpy as np


def setup_camera():
    """Initializes the webcam and sets properties."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def detect_colors(img, contour):
    """Detect colors within a specific contour."""
    mask = np.zeros_like(
        img)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    mean = cv2.mean(img, mask=mask[:, :, :1])

    black = (30, 30, 30)
    white = (220, 220, 220)
    green = (0, 128, 0)
    red = (0, 0, 255)

    b, g, r = mean[:3]

    if b < black[0] and g < black[1] and r < black[2]:
        return 'black', 5.00
    elif b > white[0] and g > white[1] and r > white[2]:
        return 'white', 0.10
    elif g > green[1] and b < green[0] and r < green[2]:
        return 'green', 1.00
    elif r > red[2] and g < red[1] and b < red[0]:
        return 'red', 0.20
    else:
        return 'unknown', 0.00


def process_frame(frame):
    """Process the frame to detect and count colored objects."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def main():
    cap = setup_camera()
    total_value = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        contours = process_frame(frame)
        output = frame.copy()

        for contour in contours:
            color, value = detect_colors(frame, contour)
            total_value += value
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            cv2.putText(output, f"{color}: ${value}", tuple(
                contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(output, f"Total Value: ${total_value:.2f}",
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Poker Chip Counter", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
