import cv2
from ultralytics import YOLO
from tkinter import Tk, Button, filedialog, Label
import os

model_path = "C:/Users/aadip/Downloads/best (1).pt"
model = YOLO(model_path)


def draw_results(frame, results):
    """Draw bounding boxes and labels on the frame."""
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = map(int, result[:6])
        label = f"{model.names[class_id]} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process_image():
    """Run inference on a selected image."""
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path and os.path.exists(file_path):
        image = cv2.imread(file_path)
        results = model.predict(source=image, conf=0.25, save=False)
        draw_results(image, results)
        cv2.imshow("Image Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No file selected or invalid file.")


def process_live_feed():
    """Run inference on live video feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model.predict(source=frame, conf=0.25, save=False)
        draw_results(frame, results)
        cv2.imshow("Live Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Create the main GUI window."""
    root = Tk()
    root.title("YOLOv8 Inference")
    root.geometry("300x200")

    Label(root, text="YOLOv8 Inference", font=("Helvetica", 16)).pack(pady=10)

    Button(root, text="Image Inference", command=process_image, width=20).pack(pady=10)

    Button(root, text="Live Webcam Inference", command=process_live_feed, width=20).pack(pady=10)

    Button(root, text="Exit", command=root.quit, width=20).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
