import psycopg2
import numpy as np
import cv2

# Connect to DB
conn = psycopg2.connect(dbname="DEMOimg", user="postgres", password="1234")
cur = conn.cursor()

cur.execute("SELECT image_data FROM detection_record WHERE id = %s", (500,))
image_data = cur.fetchone()[0]
if not image_data:
    print("No image data found for the given ID.")
    exit(1)

print("Image data length:", len(image_data))
height, width, channels = 480, 640, 3  # Corrected shape
expected_length = height * width * channels
print("Expected length:", expected_length)
if len(image_data) != expected_length:
    print("Image data size does not match expected shape!")
    exit(1)

nparr = np.frombuffer(image_data, np.uint8)
img = nparr.reshape((height, width, channels))

# If you want to display with OpenCV (which expects BGR), convert:
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow('Image from DB', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cleanup
cur.close()
conn.close()